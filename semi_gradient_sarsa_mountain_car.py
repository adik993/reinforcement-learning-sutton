import gym
import numpy as np
import sys

from features.TileCoding import *

import plotly.offline as py
import plotly.graph_objs as go
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from joblib import Parallel, delayed
from multiprocessing import cpu_count

from utils import epsilon_prob, randargmax, Algorithm, calc_batch_size, TilingValueFunction

POSITION_MIN = -1.2
POSITION_MAX = 0.6
POSITION_GOAL = 0.5
VELOCITY_MIN = -0.07
VELOCITY_MAX = 0.07
N_TILINGS = 8
MAX_SIZE = 4096

EPSILON = 0


class ValueFunction(TilingValueFunction):
    def __init__(self, n_tilings: int, iht: IHT):
        super().__init__(n_tilings, iht)

    def scaled_values(self, state):
        position, velocity = state
        position_scale = self.n_tilings / (POSITION_MAX - POSITION_MIN)
        velocity_scale = self.n_tilings / (VELOCITY_MAX - VELOCITY_MIN)
        return [position * position_scale, velocity * velocity_scale]


class SemiGradientSarsa(Algorithm):
    def __init__(self, env: gym.Env, value_function: TilingValueFunction, alpha=0.5 / N_TILINGS, gamma=1.0,
                 epsilon=EPSILON):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = np.arange(env.action_space.n)
        self.value_function = value_function
        self.next_action = None

    def _probs(self, state):
        greedy = self.greedy_action(state)
        probs = [self._prob(action, greedy) for action in self.actions]
        return probs

    def _prob(self, action, greedy):
        return epsilon_prob(greedy, action, len(self.actions), self.epsilon)

    def action(self, state):
        if self.next_action is not None:
            return self.next_action
        else:
            return self._action(state)

    def _action(self, state):
        return np.random.choice(self.actions, p=self._probs(state))

    def greedy_action(self, state):
        array = np.array([self.value_function.estimated(state, action) for action in self.actions])
        return np.argmax(array)

    def on_new_state(self, state, action, reward, next_state, done):
        self.next_action = self._action(next_state)
        q_next = self.value_function.estimated(next_state, self.next_action)
        q = self.value_function.estimated(state, action)
        delta = reward + self.gamma * q_next - q
        update = self.alpha * delta
        self.value_function[state, action] += update
        if done:
            self.next_action = None


class Entry:
    def __init__(self, state, action, reward):
        self.state = state
        self.action = action
        self.reward = reward

    def __str__(self) -> str:
        return 'Entry(state={}, action={}, reward={})'.format(self.state, self.action, self.reward)

    def __repr__(self) -> str:
        return self.__str__()


class NStepSemiGradientSarsa(Algorithm):
    def __init__(self, env: gym.Env, value_function: TilingValueFunction, n, alpha=0.5 / N_TILINGS, gamma=1,
                 epsilon=EPSILON):
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = np.arange(env.action_space.n)
        self.value_function = value_function
        self._reset()

    def _hist_size(self):
        return self.n + 1

    def _idx(self, t):
        return t % self._hist_size()

    def _reset(self):
        self.t = 0
        self.T = sys.maxsize
        self.history = [None] * self._hist_size()

    def store(self, state, action, reward, t):
        self.history[self._idx(t)] = Entry(state, action, reward)

    def get_entry(self, t):
        return self.history[self._idx(t)]

    def _get_key(self, t):
        entry = self.get_entry(t)
        return entry.state, entry.action

    def action(self, state):
        if self.t > 0:
            return self.get_entry(self.t).action
        else:
            return self._action(state)

    def _action(self, state):
        return np.random.choice(self.actions, p=self._probs(state))

    def _probs(self, state):
        greedy = self.greedy_action(state)
        probs = [self._prob(action, greedy) for action in self.actions]
        return probs

    def _prob(self, action, greedy):
        return epsilon_prob(greedy, action, len(self.actions), self.epsilon)

    def greedy_action(self, state):
        array = np.array([self.value_function.estimated(state, action) for action in self.actions])
        return randargmax(array)

    def calc_returns(self, update_time):
        return sum([pow(self.gamma, t - update_time - 1) * self.get_entry(t).reward
                    for t in range(update_time + 1, min(update_time + self.n, self.T) + 1)])

    def on_new_state(self, state, action, reward, next_state, done):
        if self.t == 0:
            self.store(state, action, None, 0)
        if self.t < self.T:
            if done:
                next_action = None
                self.T = self.t + 1
            else:
                next_action = self._action(next_state)
            self.store(next_state, next_action, reward, self.t + 1)
        update_time = self.t - self.n + 1
        if update_time >= 0:
            key_t = self._get_key(update_time)
            key_t_plus_n = self._get_key(update_time + self.n)
            returns = self.calc_returns(update_time)
            not_last_state = update_time + self.n < self.T
            if not_last_state:
                returns += pow(self.gamma, self.n) * self.value_function.estimated(*key_t_plus_n)
            self.value_function[key_t] += self.alpha * (returns - self.value_function.estimated(*key_t))
        self.t += 1
        if done and update_time != self.T - 1:
            self.on_new_state(state, action, reward, next_state, done)
        elif done:
            self._reset()


def generate_episode(env: gym.Env, algorithm: Algorithm, render=False):
    done = False
    obs = env.reset()
    counter = 0
    while not done:
        if render:
            env.render()
        prev_obs = obs
        action = algorithm.action(obs)
        obs, reward, done, _ = env.step(action)
        algorithm.on_new_state(prev_obs, action, reward, obs, done)
        counter += 1
    return counter


def do_alpha_work(n_avg, n_episode, algorithm_supplier, alpha):
    result = np.zeros((n_episode,))
    for i in range(n_avg):
        algorithm = algorithm_supplier(alpha)
        for ep in range(n_episode):
            steps = generate_episode(algorithm_supplier.env, algorithm, render=False)
            result[ep] += steps
            print('Run: {}, alpha: {}, ep: {}, steps: {}'.format(i, alpha, ep, steps))
    return result


def perform_alpha_test(algorithm_supplier, alphas, n_avg=100, n_episode=500):
    results = {alpha: np.zeros((n_episode,)) for alpha in alphas}
    with Parallel(n_jobs=cpu_count()) as parallel:
        for alpha in alphas:
            tmp = np.sum(parallel(
                delayed(do_alpha_work)(calc_batch_size(n_avg, cpu_count(), batch_idx), n_episode, algorithm_supplier,
                                       alpha)
                for batch_idx in range(cpu_count())), axis=0)
            results[alpha] = tmp
            results[alpha] /= n_avg
    return results


def do_n_work(n_avg, n_episode, algorithm_supplier, alpha, n):
    result = np.zeros((n_episode,))
    for i in range(n_avg):
        algorithm = algorithm_supplier(alpha, n)
        for ep in range(n_episode):
            steps = generate_episode(algorithm_supplier.env, algorithm, render=False)
            result[ep] += steps
            print('Run: {:2}, n: {}, ep: {:3}, steps: {:4}'.format(i, n, ep, steps))
    return result


def perform_n_test(algorithm_supplier, params, n_avg=100, n_episode=500):
    results = {n: np.zeros((n_episode,)) for alpha, n in params}
    with Parallel(n_jobs=cpu_count()) as parallel:
        for alpha, n in params:
            tmp = np.sum(parallel(
                delayed(do_n_work)(calc_batch_size(n_avg, cpu_count(), batch_idx), n_episode, algorithm_supplier, alpha,
                                   n)
                for batch_idx in range(cpu_count())), axis=0)
            results[n] = tmp
            results[n] /= n_avg
    return results


class GimmeSarsa:
    def __init__(self, env):
        self.env = env

    def __call__(self, alpha):
        return SemiGradientSarsa(self.env, ValueFunction(N_TILINGS, IHT(MAX_SIZE)), alpha)


class GimmeNStepSarsa:
    def __init__(self, env):
        self.env = env

    def __call__(self, alpha, n):
        return NStepSemiGradientSarsa(self.env, ValueFunction(N_TILINGS, IHT(MAX_SIZE)), n, alpha)


def plot_value_function_using_plotly(value_function):
    actions = np.arange(env.action_space.n)
    position_range = np.arange(POSITION_MIN, POSITION_MAX + 0.1, 0.1)  # X
    velocity_range = np.arange(VELOCITY_MIN, VELOCITY_MAX + 0.001, 0.001)  # Y
    Z = np.zeros((len(velocity_range), len(position_range)))
    for axis0, velocity in enumerate(velocity_range):
        for axis1, position in enumerate(position_range):
            Z[axis0, axis1] = max([-value_function.estimated((position, velocity, action)) for action in actions])
    surface = go.Surface(x=position_range, y=velocity_range, z=Z)
    layout = go.Layout(scene=dict(xaxis={'title': 'Position', 'range': [POSITION_MIN, POSITION_MAX]},
                                  yaxis={'title': 'Velocity', 'range': [VELOCITY_MIN, VELOCITY_MAX]},
                                  aspectratio=dict(x=1, y=1, z=0.5)))
    py.plot(go.Figure(data=[surface], layout=layout))


def plot_value_function_using_matplotlib(value_function):
    actions = np.arange(env.action_space.n)
    position_range = np.arange(POSITION_MIN, POSITION_MAX + 0.1, 0.1)  # X
    velocity_range = np.arange(VELOCITY_MIN, VELOCITY_MAX + 0.001, 0.001)  # Y
    Z = np.zeros((len(velocity_range), len(position_range)))
    X, Y = np.meshgrid(position_range, velocity_range)
    for (axis0, axis1), _ in np.ndenumerate(Z):
        Z[axis0, axis1] = max([-value_function.estimated((X[axis0, axis1], Y[axis0, axis1], action))
                               for action in actions])

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)
    plt.show()


def plot_sarsa_steps_by_alpha(env):
    alphas = [0.1 / N_TILINGS, 0.2 / N_TILINGS, 0.5 / N_TILINGS]
    results = perform_alpha_test(env, GimmeSarsa(env), alphas)
    data = []
    for alpha, values in results.items():
        data.append(go.Scatter(y=values, name='alpha={}'.format(alpha)))
    py.plot(data)


def plot_n_step_sarsa_by_alpha_and_n(env):
    params = [(0.5 / N_TILINGS, 1), (0.2 / N_TILINGS, 8)]
    results = perform_n_test(GimmeNStepSarsa(env), params)
    data = []
    for n, values in results.items():
        data.append(go.Scatter(y=values, name='n={}'.format(n)))
    py.plot(data)


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = int(1e6)
    # plot_sarsa_steps_by_alpha(env)

    plot_n_step_sarsa_by_alpha_and_n(env)

    # value_function = ValueFunction(N_TILINGS, IHT(MAX_SIZE))
    # for i in range(100):
    #     # steps = generate_episode(env, NStepSemiGradientSarsa(env, value_function, 8, 0.5 / N_TILINGS))
    #     steps = generate_episode(env, SemiGradientSarsa(env, value_function, 0.5 / N_TILINGS))
    #     print('Ep: {:3} steps: {:3}'.format(i, steps))
    #
    # plot_value_function_using_plotly(value_function)
