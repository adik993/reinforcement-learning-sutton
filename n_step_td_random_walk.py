import sys

import numpy as np
import plotly.graph_objs as go
import plotly.offline as py

from envs.RandomWalkEnv import RandomWalk
from randomwalk import rmse
from utils import Algorithm

TRUE_VALUES = np.arange(-20, 22, 2) / 20.0


class NStepTD(Algorithm):
    def __init__(self, env: RandomWalk, n, alpha=0.1, gamma=1):
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.values = np.zeros((env.observation_space.n,))
        self.action_space = env.action_space
        self._reset()

    def action(self, state):
        return self.action_space.sample()

    def _hist_idx(self, time):
        return time % (self.n + 1)

    def get_reward(self, time):
        return self.history[self._hist_idx(time)][0]

    def get_state(self, time):
        return self.history[self._hist_idx(time)][1]

    def store(self, reward, state, t):
        self.history[self._hist_idx(t)] = (reward, state)

    def _calculate_return(self, time):
        G = 0
        for t in range(time + 1, min(time + self.n, self.T) + 1):
            G += (self.gamma ** (t - time - 1)) * self.get_reward(t)
        return G

    def _reset(self):
        self.t = 0
        self.T = sys.maxsize
        self.history = [(0, 0)] * (self.n + 1)

    def on_new_state(self, state, action, reward, next_state, done):
        if self.t == 0:
            self.store(0, state, self.t)
        if self.t < self.T:
            self.store(reward, next_state, self.t + 1)
            if done:
                self.T = self.t + 1
        update_time = self.t - self.n + 1
        if update_time >= 0:
            state_to_update = self.get_state(update_time)
            step_t_plus_1 = self.get_state(update_time + self.n)
            G = self._calculate_return(update_time)
            if update_time + self.n < self.T:
                G += (self.gamma ** self.n) * self.values[step_t_plus_1]
            self.values[state_to_update] += self.alpha * (G - self.values[state_to_update])
        self.t += 1
        if done and update_time != self.T - 1:
            self.on_new_state(state, action, reward, next_state, done)
        elif done:
            self._reset()


def generate_episode(env: RandomWalk, algorithm: Algorithm):
    done = False
    obs = env.reset()
    while not done:
        prev_obs = obs
        action = algorithm.action(prev_obs)
        obs, reward, done, aux = env.step(action)
        algorithm.on_new_state(prev_obs, action, reward, obs, done)


def perform_algorithm_eval(env, algorithm_supplier, ns, alphas, n_avg=100, n_ep=10):
    ret = np.zeros((len(ns), len(alphas)))
    for i in range(n_avg):
        print('Averaging {}:'.format(i))
        for idx_n, n in enumerate(ns):
            print('Evaluating n={}'.format(n))
            for idx_alpha, alpha in enumerate(alphas):
                algorithm = algorithm_supplier(env, n, alpha)
                for ep in range(n_ep):
                    generate_episode(env, algorithm)
                ret[idx_n][idx_alpha] += rmse(algorithm.values[1:-1], TRUE_VALUES[1:-1])
    return ret / n_avg


if __name__ == '__main__':
    env = RandomWalk(n_states=19 + 2, left_reward=-1, right_reward=1, start_position=10)
    ns = np.power(2, np.arange(10))
    alphas = np.arange(0, 1.1, 0.1)
    x = perform_algorithm_eval(env, lambda env, n, alpha: NStepTD(env, n, alpha), ns, alphas)
    data = []
    for idx_n, row in enumerate(x):
        data.append(go.Scatter(x=alphas, y=row, name='{}-step TD'.format(ns[idx_n])))
    py.plot(data)
