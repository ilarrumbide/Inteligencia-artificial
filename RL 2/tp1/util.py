


def plot_rewards(rewards, window=500):
    ma = np.convolve(rewards, np.ones(window)/window, mode="valid")
    plt.plot(ma)
    plt.xlabel(f"Episodios (window={window})")
    plt.ylabel("Reward medio")
    plt.title("Convergencia Q-Learning")
     