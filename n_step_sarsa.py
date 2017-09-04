import sys

from envs.GridWorldEnv import GridWorld, Env
from double_q_learning import epsilon_prob
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go


class Algorithm:
    def action(self, state):
        raise NotImplementedError()

    def on_new_state(self, state, action, reward, next_state, done):
        raise NotImplementedError()


class NStepSarsa(Algorithm):
    def __init__(self, env: Env, n, alpha=0.1, gamma=1, epsilon=0.1):
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = np.arange(env.action_space.n)
        obs_space = [space.n for space in env.observation_space.spaces]
        self.action_values = np.zeros(obs_space + [len(self.actions)])
        self._reset()

    def _reset(self):
        self.t = 0
        self.T = sys.maxsize
        self.states_hist = [(0, 0)] * (self.n + 1)
        self.actions_hist = [0] * (self.n + 1)
        self.rewards_hist = [0] * (self.n + 1)

    def _idx(self, time):
        return time % (self.n + 1)

    def store_action(self, action, time):
        self.actions_hist[self._idx(time)] = action

    def store_state(self, state, time):
        self.states_hist[self._idx(time)] = state

    def store_reward(self, reward, time):
        self.rewards_hist[self._idx(time)] = reward

    def get_state(self, time):
        return self.states_hist[self._idx(time)]

    def get_action(self, time):
        return self.actions_hist[self._idx(time)]

    def get_reward(self, time):
        return self.rewards_hist[self._idx(time)]

    def get_key(self, time):
        return self.get_state(time) + (self.get_action(time),)

    def action(self, state):
        greedy = self.greedy_action(state)
        probs = [epsilon_prob(greedy, action, len(self.actions), self.epsilon) for action in self.actions]
        return np.random.choice(self.actions, p=probs)

    def greedy_action(self, state):
        return np.argmax(self.action_values[state])

    def on_new_state(self, state, action, reward, next_state, done):
        if self.t == 0:
            self.store_state(state, 0)
            self.store_action(action, 0)
            self.store_reward(0, 0)
        if self.t < self.T:
            self.store_state(next_state, self.t + 1)
            self.store_reward(reward, self.t + 1)
            if done:
                self.T = self.t + 1
            else:
                self.store_action(self.action(next_state), self.t + 1)
        update_time = self.t - self.n + 1
        if update_time >= 0:
            update_key = self.get_key(update_time)
            key_t_plus_1 = self.get_key(update_time + self.n)
            returns = self.calc_returns(update_time)
            if update_time + self.n < self.T:
                returns += pow(self.gamma, self.n) * self.action_values[key_t_plus_1]
            self.action_values[update_key] += self.alpha * (returns - self.action_values[update_key])
        self.t += 1
        if done and update_time != self.T - 1:
            self.on_new_state(state, action, reward, next_state, done)
        elif done:
            self._reset()

    def calc_returns(self, update_time):
        return sum([pow(self.gamma, t - update_time - 1) * self.get_reward(t) for t in
                    range(update_time + 1, min(update_time + self.n, self.T) + 1)])


def generate_episode(env: Env, algo: Algorithm):
    done = False
    count = 0
    obs = env.reset()
    while not done:
        prev_obs = obs
        action = algo.action(prev_obs)
        obs, reward, done, _ = env.step(action)
        algo.on_new_state(prev_obs, action, reward, obs, done)
        count += 1
    return count


def perform_algo_eval(env, algo_supplier, ns, n_avg=100, n_ep=100):
    ret = np.zeros((len(ns), n_ep))
    for i in range(n_avg):
        for n_idx, n in enumerate(ns):
            print('Run: {} n={}'.format(i, n))
            algo = algo_supplier(n)
            for ep in range(n_ep):
                ret[n_idx][ep] += generate_episode(env, algo)
    return ret / n_avg


if __name__ == '__main__':
    env = GridWorld()
    ns = np.power(2, np.arange(4))
    ret = perform_algo_eval(env, lambda n: NStepSarsa(env, n), ns)

    data = []
    for idx, row in enumerate(ret):
        data.append(go.Scatter(y=row, name='{}-step Sarsa'.format(ns[idx])))
    py.plot(data)
