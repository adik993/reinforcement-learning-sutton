from gym import Env

from double_q_learning import Algorithm
from envs.CliffWalkingEnv import minmax
from envs.RandomWalkEnv import RandomWalk
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go

N_AGGREGATE = 100
N_STATES = 1000
MAX_STEP = 100


def find_true_values():
    new = np.arange(-N_STATES - 1, N_STATES + 3, 2) / (N_STATES + 1)
    while True:
        old = new.copy()
        for state in range(1, N_STATES + 1):
            new[state] = 0
            for action in [-1, 1]:
                for step in range(1, MAX_STEP + 1):
                    step *= action
                    next_state = minmax(state + step, 0, N_STATES + 1)
                    prob = 1 / (MAX_STEP * 2)
                    new[state] += prob * (0 + new[next_state])
        error = np.abs(np.sum(old - new))
        print('Ture values error:', error)
        if error < 1e-2:
            break
    return new


TRUE_VALUES = find_true_values()


class State:
    def __init__(self, state, reward):
        self.state = state
        self.reward = reward


class EpisodeAlgorithm:
    def action(self, state):
        raise NotImplementedError()

    def on_new_episode(self, history):
        raise NotImplementedError()


class ValueFunction:
    def __init__(self, shape, aggregation=N_AGGREGATE):
        self.value = np.zeros([s // aggregation for s in shape])
        self.aggregation = aggregation

    def _idx(self, item):
        return (item - 1) // self.aggregation

    def __getitem__(self, item):
        return self.value[self._idx(item)]

    def __setitem__(self, key, value):
        self.value[self._idx(key)] = value


class GradientMonteCarlo(EpisodeAlgorithm):
    def __init__(self, env: Env, alpha=2e-5):
        self.alpha = alpha
        self.actions = np.arange(env.action_space.n)
        self.state_value = ValueFunction((N_STATES,), aggregation=N_AGGREGATE)

    def action(self, state):
        return np.random.choice(self.actions)

    def on_new_episode(self, history):
        # Since intermediate rewards are 0 G_t is equal to reward of terminal state
        reward = history[-1].reward
        for state in history[:-1]:
            self.state_value[state.state] += self.alpha * (reward - self.state_value[state.state])


class SemiGradientTD(Algorithm):
    def __init__(self, env: Env, alpha=2e-4, gamma=1):
        self.alpha = alpha
        self.gamma = gamma
        self.actions = np.arange(env.action_space.n)
        self.state_value = ValueFunction((N_STATES,), aggregation=N_AGGREGATE)

    def action(self, state):
        return np.random.choice(self.actions)

    def on_new_state(self, state, action, reward, next_state, done):
        value_next = 0 if done else self.state_value[next_state]
        self.state_value[state] += self.alpha * (
            reward + self.gamma * value_next - self.state_value[state])


def generate_episode(env: Env, algorithm: EpisodeAlgorithm):
    history = []
    done = False
    obs = env.reset()
    while not done:
        prev_obs = obs
        action = algorithm.action(prev_obs)
        obs, reward, done, aux = env.step(action)
        history.append(State(prev_obs, reward))
    algorithm.on_new_episode(history)
    return history


def generate_episode_td(env: Env, algorithm: Algorithm):
    done = False
    obs = env.reset()
    while not done:
        prev_obs = obs
        action = algorithm.action(prev_obs)
        obs, reward, done, aux = env.step(action)
        algorithm.on_new_state(prev_obs, action, reward, obs, done)


if __name__ == '__main__':
    env = RandomWalk(n_states=N_STATES + 2, left_reward=-1, right_reward=1, start_position=None, max_step=MAX_STEP)
    monte_carlo = GradientMonteCarlo(env)
    for i in range(int(1e5)):
        print('Episode:', i)
        generate_episode(env, monte_carlo)

    td = SemiGradientTD(env)
    for i in range(int(1e5)):
        print('Episode:', i)
        generate_episode_td(env, td)

    data = [go.Scatter(y=monte_carlo.state_value.value.repeat(N_AGGREGATE), name='Gradient Monte Carlo'),
            go.Scatter(y=td.state_value.value.repeat(N_AGGREGATE), name='Semi Gradient TD'),
            go.Scatter(y=TRUE_VALUES, name='True Values')]
    py.plot(data)
