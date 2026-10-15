"""The Q-network and its experience replay buffer."""

import random
from collections import deque, namedtuple

import torch
import torch.nn as nn

from . import config as cfg

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


class QNetwork(nn.Module):
    """state (11,) -> Q-value per action (3,). Small on purpose: this
    problem doesn't need depth, and a small net trains fast on CPU."""

    def __init__(self, state_size=cfg.STATE_SIZE, action_size=cfg.ACTION_SIZE,
                 hidden_size=cfg.HIDDEN_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Uniform-random experience replay - breaks the correlation between
    consecutive frames that would otherwise destabilize training."""

    def __init__(self, capacity=cfg.REPLAY_CAPACITY):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)
