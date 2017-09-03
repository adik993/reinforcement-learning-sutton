from collections import defaultdict, Counter

from envs.DoubleQLearningEnv import DoubleQLearningEnv
import numpy as np

import plotly.offline as py
import plotly.graph_objs as go


def randomargmax(d, key=None):
    k_max = max(d, key=key)
    return np.random.choice([k for k, v in d.items() if d[k_max] == v])


def epsilon_prob(greedy, action, n_actions, epsilon):
    if greedy == action:
        return epsilon_greedy_prob(n_actions, epsilon)
    else:
        return epsilon_explore_prob(n_actions, epsilon)


def epsilon_greedy_prob(n_actions, epsilon):
    return 1 - epsilon + epsilon / n_actions


def epsilon_explore_prob(n_actions, epsilon):
    return epsilon / n_actions


class Algorithm:
    def action(self, state):
        raise NotImplementedError()

    def on_new_state(self, state, action, reward, next_state, done):
        raise NotImplementedError()


class QLearning(Algorithm):
    def __init__(self, env, alpha=0.1, gamma=1, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = defaultdict(dict)
        for state in range(env.observation_space.n):
            actions = env.available_actions(state)
            for action in actions:
                self.q[state][action] = .0

    def action(self, state):
        greedy = self.greedy_action(state)
        actions = list(self.q[state].keys())
        probs = [epsilon_prob(greedy, action, len(actions), self.epsilon) for action in actions]
        return np.random.choice(actions, p=probs)

    def greedy_action(self, state):
        tmp = self.q[state]
        return randomargmax(tmp, key=tmp.get)

    def greedy_value(self, state):
        return self.q[state][self.greedy_action(state)]

    def on_new_state(self, state, action, reward, next_state, done):
        greedy_value = self.greedy_value(next_state) if not done else 0
        delta = self.gamma * greedy_value - self.q[state][action]
        self.q[state][action] += self.alpha * (reward + delta)


class DoubleQLearning(Algorithm):
    def __init__(self, env, alpha=0.1, gamma=1, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q1 = defaultdict(Counter)
        self.q2 = defaultdict(Counter)
        for state in range(env.observation_space.n):
            actions = env.available_actions(state)
            for action in actions:
                self.q1[state][action] = .0
                self.q2[state][action] = .0

    def action(self, state):
        greedy = self.greedy_action(state)
        actions = list(self.q1[state].keys())
        prob = [epsilon_prob(greedy, action, len(actions), self.epsilon) for action in actions]
        return np.random.choice(actions, p=prob)

    def greedy_action(self, state):
        tmp = Counter(self.q1[state])
        tmp.update(self.q2[state])
        return randomargmax(tmp, key=tmp.get)

    def greedy_action_q(self, q, state):
        return randomargmax(q[state], key=q[state].get)

    def on_new_state(self, state, action, reward, next_state, done):
        q1, q2 = (self.q1, self.q2) if np.random.rand() < 0.5 else (self.q2, self.q1)
        q1_greedy_action = 0 if done else self.greedy_action_q(q1, next_state)
        q2_value = 0 if done else q2[next_state][q1_greedy_action]
        delta = self.gamma * q2_value - q1[state][action]
        q1[state][action] += self.alpha * (reward + delta)


def generate_episode(env: DoubleQLearningEnv, algorithm: Algorithm):
    done = False
    obs = env.reset()
    while not done:
        prev_obs = obs
        action = algorithm.action(prev_obs)
        # env.render()
        obs, reward, done, _ = env.step(action)
        # print('Reward:', reward)
        algorithm.on_new_state(prev_obs, action, reward, obs, done)
    return obs


def perform_algorithm_eval(env, algorithm_supplier, n_episodes=300, n_avg=10000):
    ret = np.zeros((n_episodes,))
    for i in range(n_avg):
        print('Averaging:', i)
        algorithm = algorithm_supplier(env)
        for ep in range(n_episodes):
            ret[ep] += generate_episode(env, algorithm) == DoubleQLearningEnv.POS_TERM_LEFT
    return ret / n_avg


if __name__ == '__main__':
    env = DoubleQLearningEnv()
    q = perform_algorithm_eval(env, QLearning)
    double_q = perform_algorithm_eval(env, DoubleQLearning)
    data = [go.Scatter(y=q, name='Q-Learning'), go.Scatter(y=double_q, name='Double Q-Learning')]
    py.plot(data)
