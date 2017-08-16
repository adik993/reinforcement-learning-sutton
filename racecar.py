from collections import defaultdict

from envs.RaceCarEnv import RaceCarEnv, ACTIONS, RaceDiscrete
import numpy as np
import plotly.offline as py
from plotly import tools
import plotly.graph_objs as go

from log import make_logger

log = make_logger(__name__)

EPSILON = 0.1


class State:
    def __init__(self, axis0, axis1, vel_u, vel_r, action, reward):
        self.axis0 = axis0
        self.axis1 = axis1
        self.vel_u = vel_u
        self.vel_r = vel_r
        self.action = action
        self.reward = reward

    def __repr__(self) -> str:
        return 'State(position={}, velocity={}, action={}, reward={})' \
            .format((self.axis0, self.axis1), (self.vel_u, self.vel_r), ACTIONS[self.action], self.reward)

    def __str__(self) -> str:
        return self.__repr__()

    def to_policy_key(self):
        return self.axis0, self.axis1, self.vel_u, self.vel_r

    def to_action_value_key(self):
        return (*self.to_policy_key(), self.action)


def generate_episode(env, policy, start=None):
    history = []
    done = False
    obs = env.reset()
    if start is not None:
        env.axis0 = start[0]
        env.axis1 = start[1]
        obs = env._observations()
    while not done:
        prev_obs = obs
        action = get_action(policy, prev_obs)
        obs, reward, done, aux = env.step(action)
        history.append(State(*prev_obs, action, reward))
    return history


def get_action(policy, obs):
    vel = obs[2:]
    greedy = policy[obs]
    if np.random.rand() < EPSILON:
        random = [action for action in RaceDiscrete.legal_actions(vel) if action != greedy]
        return np.random.choice(random)
    else:
        return greedy

def policy_improvement(policy, action_values: np.ndarray, history):
    new_policy = policy.copy()
    for state in history:
        key = state.to_policy_key()
        greedy = np.argwhere(action_values[key] == np.nanmax(action_values[key])).flatten()[0]
        new_policy[key] = greedy
    return new_policy


def make_trace(road, history):
    road = road.copy()
    for state in history:
        road[state.axis0, state.axis1] = 4
    return road


def create_initial_policy():
    policy = np.full([space.n for space in env.observation_space.spaces], ACTIONS.index((0, 0)), np.int32)
    return policy


def create_initial_action_values():
    action_values = np.zeros([space.n for space in env.observation_space.spaces] + [len(ACTIONS)])
    for axis0, axis1, vel_u, vel_r, action in np.ndindex(action_values.shape):
        if not RaceDiscrete.is_legal((vel_u, vel_r), ACTIONS[action]):
            action_values[(axis0, axis1, vel_u, vel_r, action)] = np.nan
    return action_values


if __name__ == '__main__':
    env = RaceCarEnv('maps/race2.txt')
    policy = create_initial_policy()
    action_values = create_initial_action_values()
    returns = defaultdict(lambda: 0.0)
    counts = defaultdict(lambda: 0)
    for i in range(int(5e5)):
        history = generate_episode(env, policy)
        log.debug('Episode no {} last_reward={}: {}'.format(i, history[-1].reward, history))
        for i, state in enumerate(history):
            key = state.to_action_value_key()
            new = np.mean([s.reward for s in history[i:]])
            counts[key] += 1
            returns[key] += (new - returns[key]) / counts[key]
            action_values[key] = np.mean(returns[key])
        policy = policy_improvement(policy, action_values, history)

    starts = env.get_starts()
    cols = 4
    rows = int(np.ceil(len(starts) / cols))
    data = []
    titles = []
    for start in starts:
        log.debug('Generating plot starting from {}'.format(start))
        episode = generate_episode(env, policy, start)
        trace = make_trace(env.road, episode)
        data.append(go.Heatmap(z=trace))
        titles.append('Win' if episode[-1].reward > 0 else 'Lose')

    fig = tools.make_subplots(rows, cols, subplot_titles=titles)
    for i, trace in enumerate(data):
        print('Adding at ({}, {})'.format((i // cols) + 1, (i % cols) + 1))
        fig.append_trace(trace, (i // cols) + 1, (i % cols) + 1)
    py.plot(fig)
