"""Deep Q-learning utilities for the Buffalo board game.

This module provides tools for encoding board states and actions,
constructing a Q-network, and sampling experience using a replay buffer.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, NamedTuple, Optional, Tuple
import random

import torch
from torch import nn
from torch.utils.data import IterableDataset

from .board import Board, Player


class QNetwork(nn.Module):
    """Simple multi-layer perceptron approximating ``Q(s, a)``.

    The network accepts a concatenated ``[state, action]`` tensor and outputs a
    scalar Q-value.
    """

    def __init__(self, state_size: int, action_size: int = 4, hidden_size: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x).squeeze(-1)


class Transition(NamedTuple):
    state: torch.Tensor
    action: torch.Tensor
    reward: float
    next_state: torch.Tensor
    done: bool


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self) -> int:
        return len(self.buffer)


class ReplayBufferDataset(IterableDataset):
    """Simple iterable dataset that samples batches from a replay buffer."""

    def __init__(self, replay_buffer: ReplayBuffer, batch_size: int):
        self.replay_buffer = replay_buffer
        self.batch_size = batch_size

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, ...]]:
        while True:
            if len(self.replay_buffer) < self.batch_size:
                continue
            batch = self.replay_buffer.sample(self.batch_size)
            states = torch.stack(batch.state)
            actions = torch.stack(batch.action)
            rewards = torch.tensor(batch.reward, dtype=torch.float32)
            next_states = torch.stack(batch.next_state)
            dones = torch.tensor(batch.done, dtype=torch.float32)
            yield states, actions, rewards, next_states, dones


def compute_reward(board: Board, player: Player) -> float:
    """Return the reward for ``player`` given the current board state."""
    winner = board.check_for_winner()
    if winner is None:
        return 0.0
    return 1.0 if winner == player else -1.0


@dataclass
class DQNAgent:
    """Minimal Deep Q-learning agent."""

    state_size: int
    action_size: int = 4
    gamma: float = 0.99
    lr: float = 1e-3
    buffer_size: int = 10000
    batch_size: int = 64

    def __post_init__(self) -> None:
        self.q_network = QNetwork(self.state_size, self.action_size)
        self.target_network = QNetwork(self.state_size, self.action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.replay_buffer = ReplayBuffer(self.buffer_size)

    def remember(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self) -> Optional[float]:
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.stack(batch.state)
        actions = torch.stack(batch.action)
        rewards = torch.tensor(batch.reward, dtype=torch.float32)
        next_states = torch.stack(batch.next_state)
        dones = torch.tensor(batch.done, dtype=torch.float32)

        q_values = self.q_network(states, actions)
        with torch.no_grad():
            next_q_values = self.target_network(next_states, actions)
            targets = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.functional.mse_loss(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_target(self) -> None:
        self.target_network.load_state_dict(self.q_network.state_dict())
