from schemas import Transition
from collections import deque
import random
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buf: deque[Transition] = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buf)

    def add(self, t: Transition):
        self.buf.append(t)

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        s = torch.as_tensor(np.stack([b.s for b in batch]), dtype=torch.float32)
        a = torch.as_tensor([b.a for b in batch], dtype=torch.int64)
        r = torch.as_tensor([b.r for b in batch], dtype=torch.float32)
        s2 = torch.as_tensor(
            np.stack([np.zeros_like(b.s) if b.s2 is None else b.s2 for b in batch]),
            dtype=torch.float32,
        )
        d = torch.as_tensor([b.d for b in batch], dtype=torch.float32)
        return s, a, r, s2, d
