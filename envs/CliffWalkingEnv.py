from gym import Env
from gym.spaces import Tuple, Discrete
import numpy as np


def minmax(value, low, high):
    return max(min(value, high), low)


class CliffWalking(Env):
    metadata = {'render.modes': ['human']}
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    GROUND = 0
    CLIFF = 1

    def __init__(self):
        self.action_space = Discrete(4)
        self.world = np.zeros((4, 12))
        self.world[3, 1:-1] = CliffWalking.CLIFF
        self.observation_space = Tuple((Discrete(self.world.shape[0]), Discrete(self.world.shape[1])))

    def _render(self, mode='human', close=False):
        tmp = self.world.copy()
        tmp[self.position] = 8
        print('\n'+'-'*15+'\n')
        print('Position: {}'.format(str(self.position)))
        print(tmp)

    def _reset(self):
        self.start = (3, 0)
        self.stop = (3, 11)
        self.position = self.start
        return self._observation()

    def _step(self, action):
        felt = False
        if action == CliffWalking.ACTION_UP:
            felt = self._move((-1, 0))
        elif action == CliffWalking.ACTION_DOWN:
            felt = self._move((1, 0))
        elif action == CliffWalking.ACTION_LEFT:
            felt = self._move((0, -1))
        elif action == CliffWalking.ACTION_RIGHT:
            felt = self._move((0, 1))

        done = self.position == self.stop

        return self._observation(), self._reward(felt), done, self.world

    def _reward(self, felt):
        return -100 if felt else -1

    def _move(self, by):
        axis0 = minmax(self.position[0] + by[0], 0, self.world.shape[0] - 1)
        axis1 = minmax(self.position[1] + by[1], 0, self.world.shape[1] - 1)
        felt = False
        if self.world[axis0, axis1] == CliffWalking.CLIFF:
            felt = True
            axis0, axis1 = self.start
        self.position = axis0, axis1
        return felt

    def _observation(self):
        return self.position


if __name__ == '__main__':
    env = CliffWalking()
    done = False
    obs = env.reset()
    while not done:
        env.render()
        obs, reward, done, aux = env.step(env.action_space.sample())
