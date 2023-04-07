from collections import namedtuple, deque
import random
import torch
import copy

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'step'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class OUNoise:
    """Ornstein-Uhlenbeck process.
    Taken from Udacity deep-reinforcement-learning github repository:
    https://github.com/udacity/deep-reinforcement-learning/blob/master/
    ddpg-pendulum/ddpg_agent.py
    """

    def __init__(
            self,
            size: int,
            mu: float = 0.0,
            theta: float = 0.15,
            sigma: float = 0.2,
    ):
        """Initialize parameters and noise process."""
        self.state = torch.tensor(0.0)
        self.mu = mu * torch.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self, x=None):
        """Update internal state and return it as a noise sample."""
        if x == None:
            x = self.state
        # dx = self.theta * (self.mu - x) + self.sigma * np.array(
        #     [random.random() for _ in range(len(x))]
        # )
        dx = self.theta * (self.mu - x) + self.sigma * torch.rand(len(x))
        self.state = x + dx
        return self.state