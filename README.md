# Collaboration and Competition

---

In this notebook, we solve Tennis Multi-agent environment [Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893) program.

### 1. Start the Environment

We begin by importing the necessary packages.  If the code cell below returns an error, please revisit the project instructions to double-check that you have installed [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and [NumPy](http://www.numpy.org/).

### 2. Examine the State and Action Spaces

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is considered solved when the average reward of the winning agent each episode hits 0.5 over 100 consecutive episodes.

### Instructions

The file structure

```
└── submission
    ├── model.py             # Model used
    ├── ddpg_agent.py         # DQN agent
    ├── solved_checkpoint_0.pth   # Weights for agent 1
    ├── solved_checkpoint_1.pth   # Weights for agent 2
```

`ddpg_train_agents` function in the `Solution.ipynb`  notebook does the required training that generates the solved_checkpoint_0.pth and solved_checkpoint_1.pth files

