# main.py
from schemas import StateSchema
from query_router_env import QueryRouterEnv
from train import train,plot_rewards
from evaluation import evaluate_agent, evaluate_fixed

def main():
    schema = StateSchema()
    env    = QueryRouterEnv(schema=schema)

    # 1) entrenamiento
    agent, rewards = train(env, schema, episodes=300)
    plot_rewards(rewards)

    # 2) evaluación comparativa
    mean_agent   = evaluate_agent(env, agent, schema, episodes=2_000)
    mean_gpt4o   = evaluate_fixed(env, action_idx=1, episodes=2_000)  # baseline

    print(f"\nAverage reward (agent)   : {mean_agent: .3f}")
    print(f"Average reward (GPT-4o)   : {mean_gpt4o: .3f}")

if __name__ == "__main__":
    main()
