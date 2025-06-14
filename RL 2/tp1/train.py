from __future__ import annotations
import math
import numpy as np
import torch
from schemas import StateSchema, Transition
from query_router_env import QueryRouterEnv
from replay_buffer import ReplayBuffer
from agent import DoubleDQNAgent
import matplotlib.pyplot as plt


def one_hot(state: np.ndarray) -> np.ndarray:
    """One-hot encode (ctx, tools, resp) → 10-D vector."""
    ctx, tools, resp = state
    vec = np.zeros(10, dtype=np.float32)
    vec[ctx] = 1.0  # 0-2
    vec[3 + tools] = 1.0  # 3-6
    vec[7 + resp] = 1.0  # 7-9
    return vec

def plot_rewards(rewards, window=100, title="Training curve"):
    """Plot raw rewards and moving average."""
    def moving_avg(x, k):
        return np.convolve(x, np.ones(k) / k, mode="valid")

    plt.figure(figsize=(8, 4))
    plt.plot(rewards, label="Reward")
    if len(rewards) >= window:
        ma = moving_avg(rewards, window)
        plt.plot(np.arange(len(rewards) - window + 1) + window - 1, ma,
                 label=f"Moving-avg({window})")
    plt.xlabel("Episode")
    plt.ylabel("Total reward")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


def train(
    env: QueryRouterEnv,
    schema: StateSchema,
    episodes: int = 300,
    warmup: int = 300,
    batch_size: int = 64,
    eps_start: float = 1.0,
    eps_end: float = 0.05,
    eps_decay: float = 1_000.0,
):
    buffer = ReplayBuffer(50_000)
    agent = DoubleDQNAgent(
        state_dim=10,
        n_actions=3,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    rewards, eps, steps = [], eps_start, 0
    for ep in range(episodes):
        s, _ = env.reset()
        ep_reward = 0.0
        done = False

        while not done:
            s_vec = one_hot(s)
            a = agent.act(s_vec, eps)
            s2, r, done, _, _ = env.step(a)
            s2_vec = None if s2 is None else one_hot(s2)

            buffer.add(Transition(s_vec, a, r, s2_vec, done))
            ep_reward += r
            steps += 1

            # linear ε-decay
            eps = eps_end + (eps_start - eps_end) * math.exp(-steps / eps_decay)

            if len(buffer) >= warmup and len(buffer) >= batch_size:
                agent.update(buffer.sample(batch_size))

            s = s2

        rewards.append(ep_reward)
        if (ep + 1) % 100 == 0:
            mean_last_100 = np.mean(rewards[-100:])
            print(
                f"Episode {ep+1}/{episodes} | reward(avg100) = {mean_last_100:.3f} | ε = {eps:.3f}"
            )

    return agent, rewards 