from collections import defaultdict

from gym import Env, Space
from gym.spaces import Discrete

import numpy as np


class StatefullActionSpace(Space):
    def __init__(self, env):
        self.env = env

    def contains(self, x):
        return x in env.currently_available_actions()

    def sample(self):
        return np.random.choice(env.currently_available_actions())


class KeyDefaultDict(defaultdict):
    def __missing__(self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = self.default_factory(key)
        return self[key]


class Arc:
    def __init__(self, to_node, value_f):
        self.to_node = to_node
        self.value_f = value_f

    def value(self):
        return self.value_f()

    def __str__(self):
        return '--({:+.4f})-> {}'.format(self.value, self.to_node.id)

    def __repr__(self):
        return 'Arc({}, {})'.format(repr(self.to_node), repr(self.value))


class Node:
    def __init__(self, id):
        self.id = id
        self.arcs = []

    def add_arc(self, to_node, value):
        self.arcs.append(Arc(to_node, value))

    def follow(self, arc):
        return self.arcs[arc]

    def __getitem__(self, item):
        return self.follow(item)

    def __len__(self):
        return len(self.arcs)

    def __str__(self):
        value = 'Node {}:\n'.format(self.id)
        for arc in self.arcs:
            value += '\t{}\n'.format(arc)
        return value

    def __repr__(self):
        return 'Node({})'.format(self.id)


class Graph:
    def __init__(self):
        self.nodes = KeyDefaultDict(lambda key: Node(key))

    def add_arc(self, from_id, to_id, value):
        from_node = self.nodes[from_id]
        to_node = self.nodes[to_id]
        from_node.add_arc(to_node, value)

    def available_actions(self, from_id):
        return len(self.nodes[from_id])

    def __getitem__(self, item):
        return self.nodes[item]

    def __str__(self):
        value = ''
        for node in self.nodes.values():
            value += str(node)
        return value


class DoubleQLearningEnv(Env):
    metadata = {'render.modes': ['human']}
    POS_TERM_LEFT = 0
    POS_B = 1
    POS_START = 2
    POS_TERM_RIGHT = 3

    def __init__(self):
        self.observation_space = Discrete(4)
        self.action_space = StatefullActionSpace(self)
        self.mdp = self.prepare_mdp()
        self._reset()

    def prepare_mdp(self):
        mdp = Graph()
        mdp.add_arc(DoubleQLearningEnv.POS_START, DoubleQLearningEnv.POS_TERM_RIGHT, lambda: 0)
        mdp.add_arc(DoubleQLearningEnv.POS_START, DoubleQLearningEnv.POS_B, lambda: 0)
        for _ in range(10):
            mdp.add_arc(DoubleQLearningEnv.POS_B, DoubleQLearningEnv.POS_TERM_LEFT, lambda: np.random.normal(-.1, 1))
        return mdp

    def currently_available_actions(self):
        return self.available_actions(self.pos)

    def available_actions(self, state):
        return np.arange(len(self.mdp[state]))

    def _render(self, mode='human', close=False):
        print('Imma here: {}'.format(self.pos))

    def _reset(self):
        self.pos = DoubleQLearningEnv.POS_START
        return self._obs()

    def _obs(self):
        return self.pos

    def _step(self, action):
        arc = self.mdp[self.pos][action]
        self.pos = arc.to_node.id
        reward = arc.value()
        done = self.is_terminal(self.pos)
        return self._obs(), reward, done, self.mdp

    def is_terminal(self, pos):
        return pos == DoubleQLearningEnv.POS_TERM_LEFT or pos == DoubleQLearningEnv.POS_TERM_RIGHT


if __name__ == '__main__':
    env = DoubleQLearningEnv()
    print(env.mdp)
    done = False
    obs = env.reset()
    while not done:
        env.render()
        obs, reward, done, aux = env.step(env.action_space.sample())
        print('Reward:', reward)
