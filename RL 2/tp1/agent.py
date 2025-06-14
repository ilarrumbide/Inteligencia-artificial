from __future__ import annotations

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dueling_mlp import DuelingMLP


class DoubleDQNAgent:
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        device: str = "cpu",
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        self.online = DuelingMLP(state_dim, n_actions).to(self.device)
        self.target = DuelingMLP(state_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()

        self.opt = optim.Adam(self.online.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def act(self, state: np.ndarray, eps: float):
        if random.random() < eps:
            return random.randrange(self.n_actions)
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q = self.online(s)
        return int(q.argmax(dim=1).item())

    def update(self, batch, clip_grad: float | None = 1.0):
        s, a, r, s2, d = [x.to(self.device) for x in batch]

        # Q(s,a)   (gather indexes)
        q_sa = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN target
        with torch.no_grad():
            a_sel = self.online(s2).argmax(dim=1)
            q_next = self.target(s2).gather(1, a_sel.unsqueeze(1)).squeeze(1)
            y = r + self.gamma * (1 - d) * q_next

        loss = self.loss_fn(q_sa, y)

        self.opt.zero_grad()
        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(self.online.parameters(), clip_grad)
        self.opt.step()

        # Soft update
        with torch.no_grad():
            for t, o in zip(self.target.parameters(), self.online.parameters()):
                t.data.mul_(1 - self.tau).add_(self.tau * o.data)