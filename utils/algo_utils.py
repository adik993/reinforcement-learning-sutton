import itertools
from math import ceil

from gym import Env

from features.TileCoding import tiles
import numpy as np


def randomargmax(d, key=None):
    k_max = max(d, key=key)
    return np.random.choice([k for k, v in d.items() if d[k_max] == v])


def randargmax(b, **kw):
    """ a random tie-breaking argmax"""
    return np.argmax(np.random.random(b.shape) * (b == b.max()), **kw)


def epsilon_probs(greedy, actions, epsilon):
    return [epsilon_prob(greedy, action, len(actions), epsilon) for action in actions]


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


def generate_episode(env: Env, algorithm: Algorithm, render=False, print_step=False):
    done = False
    obs = env.reset()
    counter = 0
    while not done:
        if print_step:
            print('Step:', counter)
        if render:
            env.render()
        prev_obs = obs
        action = algorithm.action(obs)
        obs, reward, done, _ = env.step(action)
        algorithm.on_new_state(prev_obs, action, reward, obs, done)
        counter += 1
    return counter


class TilingValueFunction:
    ALL = slice(None, None, None)

    def __init__(self, n_tilings, iht):
        self.iht = iht
        self.n_tilings = n_tilings
        self.weights = np.zeros(iht.size)

    def scaled_values(self, state):
        raise NotImplementedError('Implement me and return scaled values from state')

    def _idx(self, state, action):
        if self.is_all_slice(state) and self.is_all_slice(action):
            return TilingValueFunction.ALL
        else:
            return tiles(self.iht, self.n_tilings,
                         self.scaled_values(state),
                         [action])

    def is_all_slice(self, item):
        return isinstance(item, slice) and item == TilingValueFunction.ALL

    def x(self, state, action):
        x = np.zeros(self.weights.shape)
        x[self._idx(state, action)] = 1
        return x

    def __getitem__(self, item):
        state, action = item
        return self.weights[self._idx(state, action)]

    def estimated(self, state, action):
        return self[state, action].sum()

    def __setitem__(self, key, value):
        state, action = key
        self.weights[self._idx(state, action)] = value

    def to_policy(self, actions, *args):
        policy = np.zeros([len(arg) for arg in args])
        for state in itertools.product(*[list(arg) for arg in args]):
            policy[state] = np.argmax([self.estimated(state, action) for action in actions])
        return policy

    def to_value(self, actions, *args):
        value = np.zeros([len(arg) for arg in args])
        for state in itertools.product(*[list(arg) for arg in args]):
            value[state] = np.max([self.estimated(state, action) for action in actions])
        return value


class TilingFunctionCreator:
    def create(self):
        raise NotImplementedError('Implement this method and return subclass of TilingValueFunction')
