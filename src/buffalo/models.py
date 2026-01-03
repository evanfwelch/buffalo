"""Neural network models for Buffalo agents."""

from __future__ import annotations

import torch
from torch import nn


class BuffaloQNetwork(nn.Module):
    """MLP that scores a (state, action) pair with a scalar Q-value."""

    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None) -> torch.Tensor:
        if action is None:
            x = state
        else:
            if state.dim() == 1:
                state = state.unsqueeze(0)
            if action.dim() == 1:
                action = action.unsqueeze(0)
            x = torch.cat([state, action], dim=-1)
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x.squeeze(-1)
