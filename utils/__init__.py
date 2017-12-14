import numpy as np
from math import ceil

def randomargmax(d, key=None):
    k_max = max(d, key=key)
    return np.random.choice([k for k, v in d.items() if d[k_max] == v])


def randargmax(b, **kw):
    """ a random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)


def epsilon_prob(greedy, action, n_actions, epsilon):
    if greedy == action:
        return epsilon_greedy_prob(n_actions, epsilon)
    else:
        return epsilon_explore_prob(n_actions, epsilon)


def epsilon_greedy_prob(n_actions, epsilon):
    return 1 - epsilon + epsilon / n_actions


def epsilon_explore_prob(n_actions, epsilon):
    return epsilon / n_actions


def calc_batch_size(size, n_batches, batch_idx):
    return max(0, min(size - batch_idx * ceil(size / n_batches), ceil(size / n_batches)))

class Algorithm:
    def action(self, state):
        raise NotImplementedError()

    def on_new_state(self, state, action, reward, next_state, done):
        raise NotImplementedError()


class EpisodeAlgorithm:
    def action(self, state):
        raise NotImplementedError()

    def on_new_episode(self, history):
        raise NotImplementedError()
