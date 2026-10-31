"""DQNAgent: ties the Q-network, target network, and replay buffer
together into epsilon-greedy action selection + a training step.

Two standard stabilizers on top of vanilla Q-learning, both load-bearing
for convergence rather than optional polish:
  - a separate target network, hard-synced every TARGET_SYNC_EVERY_STEPS,
    so the TD target isn't chasing a policy that updates every step
  - experience replay (model.ReplayBuffer), so updates are drawn from a
    shuffled window of past transitions instead of the correlated frames
    that would arrive with pure online learning
"""

import os
import random

import torch
import torch.nn as nn
import torch.optim as optim

from . import config as cfg
from .model import QNetwork, ReplayBuffer


class DQNAgent:
    def __init__(self, device=None):
        self.device = device or torch.device("cpu")
        self.policy_net = QNetwork().to(self.device)
        self.target_net = QNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.LEARNING_RATE)
        self.loss_fn = nn.SmoothL1Loss()  # Huber loss - less sensitive to reward outliers than MSE
        self.replay = ReplayBuffer()

        self.train_steps = 0

    def epsilon_for_episode(self, episode):
        """Linear decay from EPSILON_START to EPSILON_END over
        EPSILON_DECAY_EPISODES, then held flat so exploration never fully
        stops (helps avoid getting stuck in a locally-fine but bad habit)."""
        progress = min(1.0, episode / cfg.EPSILON_DECAY_EPISODES)
        return cfg.EPSILON_START + progress * (cfg.EPSILON_END - cfg.EPSILON_START)

    def act(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randrange(cfg.ACTION_SIZE)
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_t)
            return int(torch.argmax(q_values, dim=1).item())

    def remember(self, state, action, reward, next_state, done):
        self.replay.push(state, action, reward, next_state, done)

    def learn_step(self):
        """One gradient step on a sampled batch. Returns the loss, or None
        if there isn't enough replay history yet to sample from."""
        if len(self.replay) < max(cfg.MIN_REPLAY_BEFORE_TRAIN, cfg.BATCH_SIZE):
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(cfg.BATCH_SIZE)
        states, actions = states.to(self.device), actions.to(self.device)
        rewards, next_states, dones = rewards.to(self.device), next_states.to(self.device), dones.to(self.device)

        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1, keepdim=True)[0]
            target = rewards + cfg.GAMMA * max_next_q * (1.0 - dones)

        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % cfg.TARGET_SYNC_EVERY_STEPS == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_state_dict": self.policy_net.state_dict(),
            "train_steps": self.train_steps,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.policy_net.load_state_dict(checkpoint["policy_state_dict"])
        self.target_net.load_state_dict(checkpoint["policy_state_dict"])
        self.train_steps = checkpoint.get("train_steps", 0)
