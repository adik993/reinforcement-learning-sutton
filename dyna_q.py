import random

from double_q_learning import epsilon_prob
from envs.MazeEnv import BasicMaze, Maze
from n_step_sarsa import Algorithm
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py


def randomargmax(a: np.ndarray):
    return np.random.choice(np.flatnonzero(a == a.max()))


class DynaQ(Algorithm):
    def __init__(self, env: Maze, n, alpha=0.1, gamma=1.0, epsilon=0.1):
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

    def perform_update(self, state, action, reward, next_state):
        q = self.action_values[state][action]
        delta = reward + self.gamma * self.action_values[next_state].max() - q
        self.action_values[state][action] += self.alpha * delta

    def on_new_state(self, state, action, reward, next_state, done):
        self.perform_update(state, action, reward, next_state)
        self.store_visit(state, action, reward, next_state)
        for _ in range(self.n):
            (s, a), (r, n_s) = random.choice(list(self.model.items()))
            self.perform_update(s, a, r, n_s)


def generate_episode(env: Maze, algorithm: Algorithm, time_steps=None):
    done = False
    time_step = 0
    obs = env.reset()
    while not done and (time_steps is None or time_steps < time_step):
        prev_obs = obs
        action = algorithm.action(obs)
        obs, reward, done, _ = env.step(action)
        algorithm.on_new_state(prev_obs, action, reward, obs, done)
        time_step += 1
    return time_step


def perform_algo_eval(env, algorithm_supplier, ns, n_ep=50, n_avg=30):
    ret = np.zeros((len(ns), n_ep,))
    for i in range(n_avg):
        for idx_n, n in enumerate(ns):
            print('run={}, n={}'.format(i, n))
            algorithm = algorithm_supplier(n)
            for ep in range(n_ep):
                ret[idx_n][ep] += generate_episode(env, algorithm)
    return ret / n_avg


if __name__ == '__main__':
    env = BasicMaze()
    ns = [0, 5, 50]
    ret = perform_algo_eval(env, lambda n: DynaQ(env, n, gamma=0.95), ns)

    data = []
    for idx, row in enumerate(ret):
        data.append(go.Scatter(y=row, name='DynaQ({})'.format(ns[idx])))
    py.plot(data)
