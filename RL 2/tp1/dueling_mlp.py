from __future__ import annotations

import torch.nn as nn
from torch import Tensor


class DuelingMLP(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.adv = nn.Linear(hidden, n_actions)

    def forward(self, x: Tensor):
        z = self.shared(x)
        v = self.value(z)
        a = self.adv(z)
        q = v + a - a.mean(dim=1, keepdim=True)
        return q
