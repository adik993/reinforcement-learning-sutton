import random

import numpy as np
import plotly.graph_objs as go
import plotly.offline as py

from double_q_learning import epsilon_prob
from envs.MazeEnv import Maze, MazeLongShort
from utils import randomargmax, Algorithm


class DynaQ(Algorithm):
    def __init__(self, env: Maze, n, alpha=0.1, gamma=0.95, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.n = n
        sizes = [space.n for space in env.observation_space.spaces]
        self.actions = np.arange(env.action_space.n)
        self.action_values = np.zeros(sizes + [len(self.actions)])
        self.model = dict()

    def store_visit(self, state, action, reward, next_state):
        self.model[(state, action)] = reward, next_state

    def action(self, state):
        greedy = self.greedy_action(state)
        probs = [epsilon_prob(greedy, action, len(self.actions), self.epsilon) for action in self.actions]
        return np.random.choice(self.actions, p=probs)

    def greedy_action(self, state):
        return randomargmax(self.action_values[state])

    def perform_update(self, state, action, reward, next_state, additional_reward=0.0):
        q = self.action_values[state][action]
        delta = (reward + additional_reward) + self.gamma * self.action_values[next_state].max() - q
        self.action_values[state][action] += self.alpha * delta

    def perform_model_learning(self):
        for _ in range(self.n):
            (s, a), (r, n_s) = random.choice(list(self.model.items()))
            self.perform_update(s, a, r, n_s)

    def on_new_state(self, state, action, reward, next_state, done):
        self.perform_update(state, action, reward, next_state)
        self.store_visit(state, action, reward, next_state)
        self.perform_model_learning()


class DynaQPlus(DynaQ):
    def __init__(self, env: Maze, n, alpha=0.1, gamma=0.95, epsilon=0.1, k=1e-4):
        super().__init__(env, n, alpha, gamma, epsilon)
        self.k = k
        self.t = 0

    def store_visit(self, state, action, reward, next_state):
        if (state, action) not in self.model:
            for a in self.actions:
                self.model[(state, a)] = 0, state, 0
        self.model[(state, action)] = reward, next_state, self.t

    def on_new_state(self, state, action, reward, next_state, done):
        super().on_new_state(state, action, reward, next_state, done)
        self.t += 1

    def perform_model_learning(self):
        for _ in range(self.n):
            (s, a), (r, n_s, tau) = random.choice(list(self.model.items()))
            additional_reward = self.k * np.sqrt(self.t - tau)
            self.perform_update(s, a, r, n_s, additional_reward)


def generate_episode(env: Maze, algorithm: Algorithm, time_steps=None):
    done = False
    time_step = 0
    rewards = None
    if time_steps:
        rewards = np.zeros((time_steps,))
    obs = env.reset()
    while not done and (time_steps is None or time_step < time_steps):
        prev_obs = obs
        action = algorithm.action(obs)
        obs, reward, done, _ = env.step(action)
        algorithm.on_new_state(prev_obs, action, reward, obs, done)
        if rewards is not None:
            rewards[time_step] = reward
        time_step += 1
    return time_step, rewards


def perform_algo_eval(env, algorithm_supplier, ns, n_ep=50, n_avg=30):
    ret = np.zeros((len(ns), n_ep,))
    for i in range(n_avg):
        for idx_n, n in enumerate(ns):
            print('run={}, n={}'.format(i, n))
            algorithm = algorithm_supplier(n)
            for ep in range(n_ep):
                count, _ = generate_episode(env, algorithm)
                ret[idx_n][ep] += count
    return ret / n_avg


def perform_algo_eval_cumsum(env, algorithm_supplier, time_steps=3000, n_avg=20):
    ret = np.zeros((time_steps,))
    for i in range(n_avg):
        print('run=', i)
        algorithm = algorithm_supplier()
        _, rewards = generate_episode(env, algorithm, time_steps)
        ret += np.cumsum(rewards)
    return ret / n_avg


if __name__ == '__main__':
    # env = BasicMaze()
    # ns = [0, 5, 50]
    # ret = perform_algo_eval(env, lambda n: DynaQ(env, n, gamma=0.95), ns)
    # for idx, row in enumerate(ret):
    #     data.append(go.Scatter(y=row, name='DynaQ({})'.format(ns[idx])))
    # py.plot(data)

    # env = MazeShortLong()
    # ret_dynaq_plus = perform_algo_eval_cumsum(env, lambda: DynaQPlus(env, 50, alpha=0.5, gamma=0.95))
    # ret_dynaq = perform_algo_eval_cumsum(env, lambda: DynaQ(env, 50, alpha=0.5, gamma=0.95))
    # data = [go.Scatter(y=ret_dynaq_plus, name='DynaQ+'), go.Scatter(y=ret_dynaq, name='DynaQ')]
    # py.plot(data)

    env = MazeLongShort()
    ret_dynaq_plus = perform_algo_eval_cumsum(env, lambda: DynaQPlus(env, 50, alpha=0.5, gamma=0.95, k=1e-3), time_steps=6000)
    ret_dynaq = perform_algo_eval_cumsum(env, lambda: DynaQ(env, 50, alpha=0.5, gamma=0.95), time_steps=6000)
    data = [go.Scatter(y=ret_dynaq_plus, name='DynaQ+'), go.Scatter(y=ret_dynaq, name='DynaQ')]
    py.plot(data)
