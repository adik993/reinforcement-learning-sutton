from gym import Env
from gym.spaces import Discrete, Tuple
import numpy as np


def inc(tuple, val):
    return tuple[0] + val, tuple[1] + val


def minmax(value, low, high):
    return max(min(value, high), low)


class WindyGridWorld(Env):
    metadata = {'render.modes': ['human']}
    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3

    ACTION_UP_LEFT = 4
    ACTION_UP_RIGHT = 5
    ACTION_DOWN_LEFT = 6
    ACTION_DOWN_RIGHT = 7
    ACTION_NO_MOVE = 8

    def __init__(self, king_moves=False, no_move=False, stochastic_wind=False):
        self.king_moves = king_moves
        self.no_move = no_move
        self.stochastic_wind = stochastic_wind
        self.action_space = Discrete(8 + self.no_move if self.king_moves else 4)
        self._reset()
        self.observation_space = Tuple((Discrete(self.size[0]), Discrete(self.size[1])))

    def _seed(self, seed=None):
        return super()._seed(seed)

    def _render(self, mode='human', close=False):
        tmp = np.chararray(inc(self.size, 2))
        tmp[:] = '#'
        tmp[1:-1, 1:-1] = ' '
        tmp[inc(self.start, 1)] = 'S'
        tmp[inc(self.stop, 1)] = 'G'
        tmp[inc(self.position, 1)] = 'O'
        print('\n' + '-' * (self.size[1] + 2) + '\n')
        print('#{:#^10}#'.format(str(self.position)))
        print('\n'.join([row.tostring().decode('utf-8') for row in tmp]))
        print('#' + ''.join([str(w) for w in self.wind]) + '#')

    def _step(self, action):
        if action == WindyGridWorld.ACTION_UP:
            self._move((-1, 0))
        elif action == WindyGridWorld.ACTION_DOWN:
            self._move((1, 0))
        elif action == WindyGridWorld.ACTION_LEFT:
            self._move((0, -1))
        elif action == WindyGridWorld.ACTION_RIGHT:
            self._move((0, 1))
        elif self.king_moves and action == WindyGridWorld.ACTION_UP_LEFT:
            self._move((-1, -1))
        elif self.king_moves and action == WindyGridWorld.ACTION_UP_RIGHT:
            self._move((-1, 1))
        elif self.king_moves and action == WindyGridWorld.ACTION_DOWN_LEFT:
            self._move((1, -1))
        elif self.king_moves and action == WindyGridWorld.ACTION_DOWN_RIGHT:
            self._move((1, 1))
        elif self.no_move and action == WindyGridWorld.ACTION_NO_MOVE:
            self._move((0, 0))
        done = self.position == self.stop
        return self._observation(), -1, done, self.wind

    def _move(self, by):
        wind = self._get_wind(self.position[1])
        axis1 = minmax(self.position[1] + by[1], 0, self.size[1] - 1)
        axis0 = self.position[0] + by[0] - wind
        axis0 = minmax(axis0, 0, self.size[0] - 1)
        self.position = axis0, axis1

    def _get_wind(self, axis1):
        additional_wind = np.random.choice([-1, 0, 1]) if self.stochastic_wind and self.wind[axis1] else 0
        return self.wind[axis1] + additional_wind

    def _observation(self):
        return self.position

    def _reset(self):
        self.size = (7, 10)
        self.start = (3, 0)
        self.stop = (3, 7)
        self.position = self.start
        self.wind = np.array([0, 0, 0, 1, 1, 1, 2, 2, 1, 0])
        return self._observation()


if __name__ == '__main__':
    env = WindyGridWorld(king_moves=True, stochastic_wind=True)
    done = False
    obs = env.reset()
    while not done:
        env.render()
        obs, reward, done, aux = env.step(env.action_space.sample())
