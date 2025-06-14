import numpy as np
def evaluate_agent(env, agent, schema, episodes=2_000):
    """Devuelve reward medio del agente (greedy, ε=0)."""
    total = 0.0
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        while not done:
            a = agent.act(schema.to_one_hot(s), eps=0.0)  # política greedy
            s, r, done, _, _ = env.step(a)
            total += r
    return total / episodes


def evaluate_fixed(env, action_idx, episodes=2_000):
    """Reward medio si siempre elegimos la misma acción (0,1,2)."""
    total = 0.0
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        while not done:
            s, r, done, _, _ = env.step(action_idx)
            total += r
    return total / episodes

# evaluator.py  (añade al final)

model_names = ["gpt-3.5-16k", "gpt-4o-128k", "gemini-32k"]

def show_policy(agent, schema):
    """Imprime la acción greedy del agente para los 36 estados posibles."""
    for ctx in range(schema.n_ctx):
        for tools in range(schema.n_tools):
            for resp in range(schema.n_resp):
                state_vec = schema.to_one_hot(np.array([ctx, tools, resp]))
                a = agent.act(state_vec, eps=0.0)          # acción greedy
                ctx_label   = schema.ctx_bins[ctx]          # small / medium / large
                tools_label = f"{tools}" if tools < 3 else "≥3"
                resp_label  = schema.resp_bins[resp]        # short / medium / long
                print(f"{ctx_label:<6} | tools {tools_label:<2} | resp {resp_label:<6} -> {model_names[a]}")

