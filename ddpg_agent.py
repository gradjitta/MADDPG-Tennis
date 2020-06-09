import numpy as np
import random
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model import DDPGNet
from OUNoise import OUNoise




BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 5e-2              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4      # how often to update the network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Ornstein–Uhlenbeck process
eps_start = 6           # Noise level start
eps_end = 0             # Noise level end
eps_decay = 250         # episodes to decay over


class DDPGAgent:
    """
    main DDPG model with
    Replay buffer
    soft update
    actor and critic steps for both the agents
    """
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, \
                 hidden_in_critic, hidden_out_critic, seed = 0, lr_actor=1.0e-3, lr_critic=1.0e-3):
        self.actor = DDPGNet(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor = True).to(device)
        self.target_actor = DDPGNet(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor = True).to(device)
        self.critic = DDPGNet(in_critic, hidden_in_actor, hidden_out_actor, 1, actor = False).to(device)
        self.target_critic = DDPGNet(in_critic, hidden_in_actor, hidden_out_actor, 1, actor = False).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.memory = ReplayBuffer(out_actor, BUFFER_SIZE, BATCH_SIZE, seed)
        self.noise = OUNoise(out_actor, scale=1.0 )
        self.action_size = out_actor
        self.eps = eps_start
        self.t_step = 0
        self.soft_update(self.actor, self.target_actor, 1.) 
        self.soft_update(self.critic, self.target_critic, 1.)
    
    def step(self, state, action, reward, next_state, done, agent_id):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA, agent_id)
    
    def act(self, state, eps = 0.0):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.actor.eval()
        with torch.no_grad():
            action_values = self.actor(state)
        self.actor.train()
        action_vals_npy = action_values.cpu().data.numpy()
        action_vals_npy += self.eps*self.noise.noise()
        
        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.clip(action_vals_npy, -1, 1)#action_values.cpu().data.numpy()
        else:
            return np.array([random.uniform(-1,1) for _ in range(self.action_size)]) #random.choice(np.arange(self.action_size))

    
    def learn(self, experiences, GAMMA, agent_id):
        """
        main learning loop
        """
        # 1) get sample from the replay buffer
        states, actions, rewards, next_states, dones = experiences
        
        # 2) Compute target q_vals
        mask_done = (1-dones).view(-1,1)
        actions_next = self.target_actor(next_states)
        
        if agent_id == 0:
            actions_next = torch.cat((actions_next, actions[:,2:].float()), dim=1)
        else:
            actions_next = torch.cat((actions[:,:2].float(), actions_next), dim=1)
        
        # self.target_actor.train()
        state_action_next = torch.cat((next_states, actions_next.float()),1)
        q_vals_next = self.target_critic(state_action_next)
        q_vals_next = rewards + GAMMA * mask_done * q_vals_next
        q_vals_next = q_vals_next.detach()
        
        # critic loss
        state_action_vec = torch.cat((states, actions.float()),1)
        q_vals = self.critic(state_action_vec)
        critic_loss = F.mse_loss(q_vals, q_vals_next)
        
        # 3) Gradient descent on target and Q
        # backward descent by default
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 4) Gradient ascent on the policy loss
        q_actions = self.actor(states)
        
        
        if agent_id == 0:
            q_actions = torch.cat((q_actions, actions[:,2:].float()), dim=1)
        else:
            q_actions = torch.cat((actions[:,:2].float(), q_actions), dim=1)
        
        sa_actor_vec = torch.cat((states, q_actions),1)
        actor_loss = -self.critic(sa_actor_vec).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 5) update both the networks
        self.soft_update(self.actor, self.target_actor, TAU) 
        self.soft_update(self.critic, self.target_critic, TAU)
        
        # Update noise value
        self.eps = self.eps - (1/eps_decay)
        if self.eps < eps_end:
            self.eps=eps_end
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
            
            

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)