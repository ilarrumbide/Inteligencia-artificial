import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(rewards, window=500):
    ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
    plt.plot(ma)
    plt.xlabel(f"Episodios (window={window})")
    plt.ylabel("Reward medio")
    plt.title("Convergencia Q-Learning")


model_names = ["gpt-3.5-16k", "gpt-4o-128k", "gemini-32k"]

def show_policy(Q, env):
    for s in range(env.observation_space.n):
        a = np.argmax(Q[s])
        ctx, tools, resp = env.decode(s)
        ctx_label   = env.ctx_bins[ctx]          # small / medium / large
        tools_label = f"{env.tool_bins[tools]}" if tools < 3 else "≥3"
        resp_label  = env.resp_bins[resp]        # short / medium / long
        print(f"{ctx_label:<6} | tools {tools_label:<2} | resp {resp_label:<6} -> {model_names[a]}")

     