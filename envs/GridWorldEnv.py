from gym import Env
import numpy as np
from gym.spaces import Tuple, Discrete

from envs.WindyGridWorldEnv import minmax


class GridWorld(Env):
    metadata = {'render.modes': ['human']}
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    def __init__(self, shape=(8, 10), start=(4, 1), end=(4, 6)):
        self.world = np.zeros(shape)
        self.start = start
        self.end = end
        self.observation_space = Tuple((
            Discrete(self.world.shape[0]),
            Discrete(self.world.shape[1])
        ))
        self.action_space = Discrete(4)
        self._reset()

    def _obs(self):
        return self.position

    def _render(self, mode='human', close=False):
        print('Position: {}'.format(self.position))
        tmp = self.world.copy()
        tmp[self.end] = 7
        tmp[self.position] = 8
        print(tmp)

    def _step(self, action):
        move = (0, 0)
        if action == GridWorld.ACTION_UP:
            move = (-1, 0)
        elif action == GridWorld.ACTION_DOWN:
            move = (1, 0)
        elif action == GridWorld.ACTION_LEFT:
            move = (0, -1)
        elif action == GridWorld.ACTION_RIGHT:
            move = (0, 1)
        self._move(move)
        done = self.position == self.end
        return self._obs(), -1, done, self.world

    def _move(self, move):
        axis0 = minmax(self.position[0] + move[0], 0, self.world.shape[0] - 1)
        axis1 = minmax(self.position[1] + move[1], 0, self.world.shape[1] - 1)
        self.position = (axis0, axis1)

    def _reset(self):
        self.position = self.start
        return self._obs()


if __name__ == '__main__':
    env = GridWorld()
    done = False
    obs = env.reset()
    while not done:
        env.render()
        obs, reward, done, aux = env.step(env.action_space.sample())
