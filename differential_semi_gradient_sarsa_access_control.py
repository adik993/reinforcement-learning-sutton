import numpy as np
from gym import Env
import plotly.offline as py
import plotly.graph_objs as go
from plotly import tools

from envs.AcessControlQueueEnv import AccessControlQueueTimeLimit, AccessControlQueue
from features.TileCoding import IHT
from utils import Algorithm, randargmax, generate_episode, epsilon_probs, TilingValueFunction

np.random.seed(7)


class ValueFunction(TilingValueFunction):
    def __init__(self, n_tilings, max_size, n_priorities, n_servers):
        super().__init__(n_tilings, IHT(max_size))
        self.n_priorities = n_priorities - 1
        self.n_servers = n_servers

    def scaled_values(self, state):
        priority, free_servers = state
        priority_scale = self.n_tilings / self.n_priorities
        server_scale = self.n_tilings / self.n_servers
        return [priority_scale * priority, server_scale * free_servers]


class DifferentialSemiGradientSarsa(Algorithm):
    def __init__(self, env: Env, value_function, alpha=0.01, beta=0.01, epsilon=0.1):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.actions = np.arange(env.action_space.n)
        self.value_function = value_function
        self._reset()

    def action(self, state):
        if self.next_action is not None:
            return self.next_action
        else:
            return self._action(state)

    def _action(self, state):
        _, free_servers = state
        if free_servers == 0:
            return AccessControlQueue.ACTION_REJECT
        greedy = self._greedy_action(state)
        probs = epsilon_probs(greedy, self.actions, self.epsilon)
        return np.random.choice(self.actions, p=probs)

    def _greedy_action(self, state):
        return randargmax(np.array([self.value_function.estimated(state, action) for action in self.actions]))

    def _reset(self):
        self.average_reward = 0
        self.next_action = None

    def on_new_state(self, state, action, reward, next_state, done):
        self.next_action = self._action(next_state)
        q_next = self.value_function.estimated(next_state, self.next_action)
        q = self.value_function.estimated(state, action)
        delta = reward - self.average_reward + q_next - q
        self.average_reward += self.beta * delta
        print('Average reward:', self.average_reward)
        self.value_function[state, action] += self.alpha * delta
        if done:
            self._reset()


if __name__ == '__main__':
    n_servers = 10
    env = AccessControlQueueTimeLimit(max_episode_steps=int(1e6), free_prob=0.06, n_servers=n_servers)
    value_function = ValueFunction(8, 2048, len(AccessControlQueue.PRIORITIES), n_servers)
    algorithm = DifferentialSemiGradientSarsa(env, value_function, alpha=0.01 / value_function.n_tilings)
    generate_episode(env, algorithm, print_step=True)

    policy = value_function.to_policy(algorithm.actions, AccessControlQueue.PRIORITIES, np.arange(n_servers + 1))
    values = value_function.to_value(algorithm.actions, AccessControlQueue.PRIORITIES, np.arange(n_servers + 1))

    fig = tools.make_subplots(rows=1, cols=2)
    fig.append_trace(go.Heatmap(z=policy,
                                x=np.arange(n_servers + 1),
                                y=AccessControlQueue.REWARDS,
                                name='Policy'), 1, 1)
    for i, row in enumerate(values):
        row[0] = value_function.estimated((i, 0), AccessControlQueue.ACTION_REJECT)
        fig.append_trace(go.Scatter(y=row, name='n={}'.format(AccessControlQueue.REWARDS[i])), 1, 2)
    fig.layout.yaxis1.autorange = 'reversed'
    py.plot(fig)
