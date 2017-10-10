import gym
import numpy as np
from features.TileCoding import *

from double_q_learning import Algorithm, epsilon_prob
import plotly.offline as py
import plotly.graph_objs as go
from joblib import Parallel, delayed
from multiprocessing import cpu_count
from math import ceil

POSITION_MIN = -1.2
POSITION_MAX = 0.6
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07
N_TILINGS = 8
MAX_SIZE = 2048


class TilingValueFunction:
    def __init__(self, n_tilings=N_TILINGS, max_size=MAX_SIZE):
        self.iht = IHT(MAX_SIZE)
        self.n_tilings = n_tilings
        self.weights = np.zeros((max_size,))
        self.position_scale = self.n_tilings / (POSITION_MAX - POSITION_MIN)
        self.velocity_scale = self.n_tilings / (VELOCITY_MAX - VELOCITY_MIN)

    def _idx(self, item):
        position, velocity, action = item
        return tiles(self.iht, self.n_tilings,
                     [self.position_scale * position, self.velocity_scale * velocity],
                     [action])

    def __getitem__(self, item):
        return self.weights[self._idx(item)]

    def estimated(self, item):
        return self[item].sum()

    def __setitem__(self, key, value):
        self.weights[self._idx(key)] = value


class SemiGradientSarsa(Algorithm):
    def __init__(self, env: gym.Env, value_function: TilingValueFunction, alpha=0.5 / N_TILINGS, gamma=1, epsilon=0.2):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = np.arange(env.action_space.n)
        self.value_function = value_function

    def _probs(self, state):
        greedy = self.greedy_action(state)
        probs = [self._prob(action, greedy) for action in self.actions]
        return probs

    def _prob(self, action, greedy):
        return epsilon_prob(greedy, action, len(self.actions), self.epsilon)

    def action(self, state):
        return np.random.choice(self.actions, p=self._probs(state))

    def greedy_action(self, state):
        array = np.array([self.value_function.estimated((*state, action)) for action in self.actions])
        return np.argmax(array)

    def on_new_state(self, state, action, reward, next_state, done):
        next_action = self.action(next_state)
        q_next = 0 if done else self.value_function.estimated((*next_state, next_action))
        q = self.value_function.estimated((*state, action))
        delta = reward + self.gamma * q_next - q
        update = self.alpha * delta
        self.value_function[(*state, action)] += update
        return next_action


def generate_episode(env: gym.Env, algorithm: Algorithm, render=False):
    done = False
    obs = env.reset()
    action = algorithm.action(obs)
    counter = 0
    while not done:
        if render:
            env.render()
        prev_obs = obs
        obs, reward, done, _ = env.step(action)
        action = algorithm.on_new_state(prev_obs, action, reward, obs, done)
        counter += 1
    return counter


def do_work(n_avg, n_episode, algorithm_supplier, alpha):
    result = np.zeros((n_episode,))
    for i in range(n_avg):
        algorithm = algorithm_supplier(alpha)
        for ep in range(n_episode):
            steps = generate_episode(env, algorithm, render=False)
            result[ep] += steps
            print('Run: {}, alpha: {}, ep: {}, steps: {}'.format(i, alpha, ep, steps))
    return result


def calc_batch_size(size, batches, batch_idx):
    return max(0, min(size - batch_idx * ceil(size / batches), ceil(size / batches)))


def perform_alpha_test(env, algorithm_supplier, alphas, n_avg=100, n_episode=500):
    results = {alpha: np.zeros((n_episode,)) for alpha in alphas}
    with Parallel(n_jobs=cpu_count()) as parallel:
        for alpha in alphas:
            tmp = np.sum(parallel(
                delayed(do_work)(calc_batch_size(n_avg, cpu_count(), batch_idx), n_episode, algorithm_supplier, alpha)
                for batch_idx in range(cpu_count())), axis=0)
            results[alpha] = tmp
            results[alpha] /= n_avg
    return results


class GimmeSarsa:
    def __init__(self, env):
        self.env = env

    def __call__(self, alpha):
        return SemiGradientSarsa(self.env, TilingValueFunction(), alpha)


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = int(1e6)
    alphas = [0.1 / N_TILINGS, 0.2 / N_TILINGS, 0.5 / N_TILINGS]
    results = perform_alpha_test(env, GimmeSarsa(env), alphas)
    data = []
    for alpha, values in results.items():
        data.append(go.Scatter(y=values, name='alpha={}'.format(alpha)))

    py.plot(data)
