def train_q(env, episodes=50_000, alpha=0.1, gamma=0.95, eps_start=1.0, eps_end=0.05):
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))
    rewards = []
    for ep in range(episodes):
        s, _ = env.reset()
        done = False
        total = 0.0
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / episodes)
        while not done:
            a = env.action_space.sample() if np.random.rand() < eps else np.argmax(Q[s])
            s2, r, done, _, _ = env.step(a)
            best_next = 0 if done else np.max(Q[s2])
            Q[s, a] += alpha * (r + gamma * best_next - Q[s, a])
            s = s2
            total += r
        rewards.append(total)
    return Q, rewards