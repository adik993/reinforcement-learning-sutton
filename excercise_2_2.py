from multiprocessing import cpu_count

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed


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


class TestCase:
    def __init__(self, levers, selector, name=None):
        self.levers = levers
        self.selector = selector
        self.name = name

    def get_name(self):
        return str(self.selector)

    def __run_test(self, bandit, selector, K=1000):
        history = np.empty(K)
        for i in range(K):
            reward = bandit.pull(selector)
            history[i] = reward
        return history

    def run(self, K, N_AVG):
        history = np.zeros(K)
        for _ in range(N_AVG):
            history += self.__run_test(Bandit(self.levers), self.selector, K)
        return history / N_AVG


if __name__ == '__main__':
    N = 10
    K = 1000
    N_AVG = 500

    tests = [TestCase(N, EpsilonGreedyActionSelector(0.0)),
             TestCase(N, EpsilonGreedyActionSelector(0.1)),
             TestCase(N, EpsilonGreedyActionSelector(0.01)),
             TestCase(N, SoftmaxActionSelector(0.1)),
             TestCase(N, SoftmaxActionSelector(0.2)),
             TestCase(N, SoftmaxActionSelector(0.4))]

    with Parallel(n_jobs=min([cpu_count(), len(tests)])) as parallel:
        results = parallel(delayed(test_case.run)(K, N_AVG) for test_case in tests)

    for test_case, history in zip(tests, results):
        plt.plot(history, label=test_case.get_name())

    plt.legend()
    plt.show()
