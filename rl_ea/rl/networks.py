from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, dims, activation=nn.ReLU, out_activation=None):
        super().__init__()
        layers = []
        for i in range(len(dims) - 2):
            layers += [nn.Linear(dims[i], dims[i + 1]), activation()]
        layers += [nn.Linear(dims[-2], dims[-1])]
        if out_activation is not None:
            layers += [out_activation()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, max_action: torch.Tensor | float = 1.0):
        super().__init__()
        self.net = MLP([state_dim, hidden_dim, hidden_dim, action_dim])
        if isinstance(max_action, (float, int)):
            max_action = torch.tensor(float(max_action))
        self.register_buffer('max_action', max_action)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        action = torch.tanh(self.net(state))
        return action * self.max_action


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.q1 = MLP([state_dim + action_dim, hidden_dim, hidden_dim, 1])
        self.q2 = MLP([state_dim + action_dim, hidden_dim, hidden_dim, 1])

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa), self.q2(sa)

    def q1_only(self, state: torch.Tensor, action: torch.Tensor):
        sa = torch.cat([state, action], dim=-1)
        return self.q1(sa)
