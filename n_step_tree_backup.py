import sys

import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
from gym import Env

from double_q_learning import epsilon_prob
from envs.GridWorldEnv import GridWorld
from n_step_sarsa import Algorithm, perform_algo_eval, NStepSarsa


class Entry:
    def __init__(self):
        self.state = 0
        self.action = 0
        self.action_value = 0
        self.delta = 0
        self.prob = 0

    def __str__(self):
        return 'Entry(state={}, action={}, action_value={}, delta={}, prob={})' \
            .format(self.state, self.action, self.action_value, self.delta, self.prob)

    def __repr__(self):
        return self.__str__()


class NStepTreeBackup(Algorithm):
    def __init__(self, env: Env, n, alpha=0.1, gamma=1, epsilon=0.1):
        self.n = n
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.actions = np.arange(env.action_space.n)
        obs_size = [space.n for space in env.observation_space.spaces]
        self.action_values = np.zeros(obs_size + [len(self.actions)])
        self._reset()

    def _reset(self):
        self.t = 0
        self.T = sys.maxsize
        self.history = [Entry()] * (self.n + 1)

    def probs(self, state):
        greedy = self.greedy_action(state)
        return [self.prob(action, greedy) for action in self.actions]

    def prob(self, action, greedy):
        return epsilon_prob(greedy, action, len(self.actions), self.epsilon)

    def action(self, state):
        if self.t > 0:
            return self.get_entry(self.t).action
        else:
            return self._action(state)

    def _action(self, state):
        return np.random.choice(self.actions, p=self.probs(state))

    def greedy_action(self, state):
        return np.argmax(self.action_values[state])

    def _idx(self, time):
        return time % (self.n + 1)

    def get_entry(self, time):
        return self.history[self._idx(time)]

    def store(self, time, entry):
        self.history[self._idx(time)] = entry

    def calc_delta_sum(self, action_taken, greedy, next_state):
        # Why it doesn't convert with filter enabled?
        # f = filter(lambda a: a != action_taken, self.actions)
        return sum(map(lambda a: self.prob(a, greedy) * self.action_values[next_state][a],
                       self.actions))

    def on_new_state(self, state, action, reward, next_state, done):
        if self.t == 0:
            new_entry = Entry()
            new_entry.state = state
            new_entry.action = action
            new_entry.action_value = self.action_values[state][action]
            self.store(self.t, new_entry)
        if self.t < self.T:
            new_entry = Entry()
            new_entry.state = next_state
            entry_t = self.get_entry(self.t)
            if done:
                self.T = self.t + 1
                entry_t.delta = reward - entry_t.action_value
            else:
                greedy = self.greedy_action(next_state)
                tmp = self.calc_delta_sum(action, greedy, next_state)
                entry_t.delta = reward + self.gamma * tmp - entry_t.action_value
                new_entry.action = self._action(next_state)
                new_entry.action_value = self.action_values[next_state][new_entry.action]
                new_entry.prob = self.prob(new_entry.action, greedy)
            self.store(self.t, entry_t)
            self.store(self.t + 1, new_entry)
        update_time = self.t - self.n + 1
        if update_time >= 0:
            e = 1
            entry_t = self.get_entry(update_time)
            returns = entry_t.action_value
            for time in range(update_time, min(update_time + self.n, self.T)):
                returns += e * self.get_entry(time).delta
                e *= self.gamma * self.get_entry(time + 1).prob
            value = self.action_values[entry_t.state][entry_t.action]
            self.action_values[entry_t.state][entry_t.action] += self.alpha * (returns - value)
        self.t += 1
        if done and update_time != self.T - 1:
            self.on_new_state(state, action, reward, next_state, done)
        elif done:
            self._reset()


if __name__ == '__main__':
    env = GridWorld()
    ns = np.power(2, np.arange(4))
    ret_tree_backup = perform_algo_eval(env, lambda n: NStepTreeBackup(env, n), ns)

    data = []
    for idx, row in enumerate(ret_tree_backup):
        data.append(go.Scatter(y=row, name='{}-step Tree Backup'.format(ns[idx])))
    py.plot(data)
