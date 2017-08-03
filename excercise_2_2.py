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

    def __repr__(self):
        return 'EpsilonGreedyActionSelector({})'.format(self.epsilon)

    def __str__(self):
        return 'epsilon={}'.format(self.epsilon)


class SoftmaxActionSelector(ActionSelector):
    def __init__(self, temperature):
        self.temperature = temperature

    def select(self, estimated_values):
        prob = self.softmax(estimated_values / self.temperature)
        return np.random.choice(np.arange(len(estimated_values)), p=prob)

    def softmax(self, x):
        return np.e ** x / sum(np.e ** x)

    def __repr__(self):
        return 'SoftmaxActionSelector({})'.format(self.temperature)

    def __str__(self):
        return 'softmax={}'.format(self.temperature)


class Bandit:
    def __init__(self, step_size):
        self.n = 10
        self.values = np.zeros(self.n)
        self.estimated_values = np.zeros(self.n)
        self.counts = np.zeros(self.n)
        self.step_size = step_size

    def pull(self, selector: ActionSelector):
        lever = selector.select(self.estimated_values)
        noise = np.random.normal(0, 0.3)
        reward = self.values[lever] + noise
        self.incremental_mean(lever, reward)
        self.random_walk()
        return reward

    def random_walk(self):
        walk = np.random.normal(0, 0.01, self.values.shape)
        if np.random.rand() < 0.045:
            idx = np.argmax(self.values)
            if walk[idx] > 0:
                walk[idx] = -walk[idx]
        self.values += walk

    def incremental_mean(self, lever, new_value):
        self.counts[lever] += 1
        if self.step_size is None:
            step_size = 1 / self.counts[lever]
        else:
            step_size = self.step_size
        self.estimated_values[lever] += step_size * (new_value - self.estimated_values[lever])


class TestCase:
    def __init__(self, step_size, selector, name=None):
        self.step_size = step_size
        self.selector = selector
        self.name = name

    def get_name(self):
        if self.name is None:
            return str(self.selector)
        else:
            return self.name

    def __run_test(self, bandit, selector, K):
        history = np.empty(K)
        for i in range(K):
            reward = bandit.pull(selector)
            history[i] = reward
        return history

    def run(self, K, N_AVG):
        history = np.zeros(K)
        for _ in range(N_AVG):
            history += self.__run_test(Bandit(self.step_size), self.selector, K)
        return history / N_AVG


if __name__ == '__main__':
    K = 10000
    N_AVG = 500
    test_case = TestCase(None, EpsilonGreedyActionSelector(0.1), name='mean')
    history = test_case.run(K, N_AVG)
    plt.plot(history, label=test_case.get_name())
    test_case = TestCase(0.1, EpsilonGreedyActionSelector(0.1), name='const')
    history = test_case.run(K, N_AVG)
    plt.plot(history, label=test_case.get_name())

    plt.legend()
    plt.show()
