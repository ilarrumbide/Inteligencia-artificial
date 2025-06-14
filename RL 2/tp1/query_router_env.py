from __future__ import annotations

import gymnasium as gym
import numpy as np
from schemas import StateSchema


class QueryRouterEnv(gym.Env):
    """Routing toy-env: choose the best LLM given context-size, #tools, resp-len."""

    metadata = {"render_modes": []}
    action_map = ["gpt-3.5-16k", "gpt-4o-128k", "gemini-32k"]

    def __init__(self, max_steps: int = 100, schema: StateSchema = StateSchema()):
        super().__init__()
        self.schema = schema
        # MultiDiscrete → explicit factorisation of state
        self.observation_space = gym.spaces.MultiDiscrete(
            [schema.n_ctx, schema.n_tools, schema.n_resp]
        )
        self.action_space = gym.spaces.Discrete(len(self.action_map))

        self.max_steps = max_steps
        self.rng = np.random.default_rng()
        self.state: np.ndarray | None = None
        self.step_cnt = 0

        # base (μ_q, σ_q, μ_lat) for ctx == small, indexed tools × resp × model
        self.base_perf = np.array(
            [
                [
                    [(0.78, 0.07, 0.25), (0.91, 0.04, 0.55), (0.83, 0.07, 0.35)],
                    [(0.72, 0.07, 0.40), (0.90, 0.04, 0.75), (0.82, 0.07, 0.55)],
                    [(0.65, 0.08, 0.55), (0.91, 0.04, 0.95), (0.84, 0.07, 0.70)],
                ],
                [
                    [(0.75, 0.08, 0.30), (0.90, 0.04, 0.60), (0.82, 0.07, 0.40)],
                    [(0.70, 0.08, 0.45), (0.90, 0.04, 0.85), (0.80, 0.07, 0.60)],
                    [(0.63, 0.09, 0.60), (0.92, 0.04, 1.05), (0.85, 0.06, 0.80)],
                ],
                [
                    [(0.73, 0.08, 0.35), (0.89, 0.04, 0.65), (0.81, 0.07, 0.45)],
                    [(0.68, 0.08, 0.50), (0.91, 0.04, 0.90), (0.82, 0.07, 0.65)],
                    [(0.60, 0.09, 0.65), (0.92, 0.04, 1.10), (0.85, 0.06, 0.85)],
                ],
                [
                    [(0.70, 0.08, 0.40), (0.88, 0.04, 0.70), (0.80, 0.07, 0.50)],
                    [(0.65, 0.08, 0.55), (0.91, 0.04, 1.00), (0.82, 0.07, 0.70)],
                    [(0.58, 0.10, 0.70), (0.92, 0.04, 1.20), (0.85, 0.06, 0.90)],
                ],
            ],
            dtype=object,
        )

    # ---------- helpers ----------
    def _adjust_for_ctx(self, ctx: int, tools: int, resp: int):
        arr = np.array(self.base_perf[tools][resp], dtype=float)

        if ctx == 1:  # medium
            arr[0, 0] -= 0.10
            arr[0, 2] += 0.20
            arr[1, 0] += 0.15
            arr[1, 2] += 0.20
            arr[2, 0] += 0.04
            arr[2, 2] += 0.20
        elif ctx == 2:  # large
            arr[0, 0] -= 0.30
            arr[0, 2] += 0.40
            arr[1, 0] += 0.25
            arr[1, 2] += 0.40
            arr[2, 0] -= 0.05
            arr[2, 2] += 0.25

        if tools == 3 or resp == 2:  # bonus for GPT-4o
            arr[1, 0] += 0.15
        return arr

    def _sample_state(self):
        while True:
            ctx = self.rng.choice(3, p=[0.55, 0.35, 0.10])
            tools = self.rng.choice(4, p=[0.45, 0.30, 0.15, 0.10])
            resp = self.rng.choice(3, p=[0.45, 0.40, 0.15])
            if not (ctx == 2 and tools == 0):
                return np.array([ctx, tools, resp], dtype=np.int32)

    # ---------- gym interface ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_cnt = 0
        self.state = self._sample_state()
        return self.state.copy(), {}

    def step(self, action: int):
        ctx, tools, resp = self.state
        # infeasible: model window too small
        if ctx == 2 and action in (0, 2):
            reward = -10.0
        else:
            μ_q, σ_q, μ_lat = self._adjust_for_ctx(ctx, tools, resp)[action]
            q = np.clip(self.rng.normal(μ_q, σ_q), 0.0, 1.0)
            lat_s = max(0.1, self.rng.normal(μ_lat, 0.05))
            reward = q - 1.0 * lat_s

        self.step_cnt += 1
        done = self.step_cnt >= self.max_steps
        self.state = None if done else self._sample_state()
        return (None if done else self.state.copy()), reward, done, False, {}

    def render(self):
        pass  # not needed
