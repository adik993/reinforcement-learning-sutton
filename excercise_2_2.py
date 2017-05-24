import matplotlib.pyplot as plt
import numpy as np


class ActionSelector:
    def select(self, estimated_values):
        raise NotImplementedError("Implement me")


class EpsilonGreedyActionSelector(ActionSelector):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select(self, estimated_values):
        if np.random.rand() < self.epsilon:
            return np.random.choice(np.arange(len(estimated_values)))
        else:
            return np.argmax(estimated_values)


class SoftmaxActionSelector(ActionSelector):
    def __init__(self, temperature):
        self.temperature = temperature

    def select(self, estimated_values):
        prob = self.softmax(estimated_values / self.temperature)
        return np.random.choice(np.arange(len(estimated_values)), p=prob)

    def softmax(self, x):
        return np.e ** x / sum(np.e ** x)


class Bandit:
    def __init__(self, n=10):
        self.n = n
        self.values = np.random.randn(self.n)
        self.estimated_values = np.zeros(self.n)
        self.counts = np.zeros(self.n)

    def pull(self, selector: ActionSelector):
        lever = selector.select(self.estimated_values)
        noise = np.random.randn()
        reward = self.values[lever] + noise
        self.incremental_mean(lever, reward)
        return reward

    def incremental_mean(self, lever, new_value):
        self.counts[lever] += 1
        self.estimated_values[lever] += (new_value - self.estimated_values[lever]) / self.counts[lever]


def run_test(bandit, selector, K=1000):
    history = np.empty((K,))
    for i in range(K):
        reward = bandit.pull(selector)
        history[i] = reward
    return history


if __name__ == '__main__':
    N = 10
    K = 1000
    N_AVG = 500
    epsilon0_greedy_history = np.zeros(K)
    epsilon001_greedy_history = np.zeros(K)
    epsilon01_greedy_history = np.zeros(K)
    softmax01_history = np.zeros(K)
    softmax02_history = np.zeros(K)
    softmax04_history = np.zeros(K)
    for i in range(500):
        bandit = Bandit(N)
        softmax01_history += run_test(bandit, SoftmaxActionSelector(0.1))
        bandit = Bandit(N)
        softmax04_history += run_test(bandit, SoftmaxActionSelector(0.2))
        bandit = Bandit(N)
        softmax02_history += run_test(bandit, SoftmaxActionSelector(0.4))
        bandit = Bandit(N)
        epsilon0_greedy_history += run_test(bandit, EpsilonGreedyActionSelector(0.0))
        bandit = Bandit(N)
        epsilon001_greedy_history += run_test(bandit, EpsilonGreedyActionSelector(0.01))
        bandit = Bandit(N)
        epsilon01_greedy_history += run_test(bandit, EpsilonGreedyActionSelector(0.1))

    plt.plot(softmax01_history / N_AVG, label='softmax=0.1')
    plt.plot(softmax02_history / N_AVG, label='softmax=0.2')
    plt.plot(softmax04_history / N_AVG, label='softmax=0.4')
    plt.plot(epsilon0_greedy_history / N_AVG, label='epsilon=0.0')
    plt.plot(epsilon001_greedy_history / N_AVG, label='epsilon=0.01')
    plt.plot(epsilon01_greedy_history / N_AVG, label='epsilon=0.1')
    plt.legend()
    plt.show()
