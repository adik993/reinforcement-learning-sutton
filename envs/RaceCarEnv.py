from enum import Enum
import os
from gym import Env
from gym.spaces.discrete import Discrete
import numpy as np

from log import make_logger

log = make_logger(__name__)


class Action(Enum):
    UP_PLUS = 0
    UP_ZERO = 1
    UP_MINUS = 2
    LEFT_PLUS = 3
    LEFT_ZERO = 4
    LEFT_MINUS = 5
    RIGHT_PLUS = 6
    RIGHT_ZERO = 7
    RIGHT_MINUS = 8

    def velocity(self):
        if self in [Action.UP_PLUS, Action.LEFT_PLUS, Action.LEFT_PLUS]:
            return 1
        elif self in [Action.UP_MINUS, Action.LEFT_MINUS, Action.LEFT_MINUS]:
            return -1
        else:
            return 0

    def is_left(self):
        return self in [Action.LEFT_MINUS, Action.LEFT_ZERO, Action.LEFT_PLUS]

    def is_up(self):
        return self in [Action.UP_MINUS, Action.UP_ZERO, Action.UP_PLUS]

    def is_right(self):
        return self in [Action.RIGHT_MINUS, Action.RIGHT_ZERO, Action.RIGHT_PLUS]

    @staticmethod
    def values():
        return list(Action)


class CellType(Enum):
    OFF = 0
    ROAD = 1
    START = 2
    STOP = 3

    @staticmethod
    def values():
        return list(CellType)


def load_race(path):
    ret = []
    with open(path) as f:
        for line in f:
            ret.append([int(x) for x in line.strip()])
    return np.array(ret, dtype=np.int32)


class RaceCarEnv(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.road_template = load_race('../maps/race1.txt')
        self.action_space = Discrete(len(Action.values()))
        self.observation_space = None  # ??? (x, y, vel_l, vel_u, vel_r)
        self.reward_range = (-5, 1)
        self._reset()

    def is_out_of_range(self, pos):
        return self.is_axis_out_of_range(pos[0], 0) or self.is_axis_out_of_range(pos[1], 1)

    def is_axis_out_of_range(self, value, axis):
        return value < 0 or value >= self.road.shape[axis]

    @staticmethod
    def add_velocity(velocity):
        return min(max(velocity, 0), 5)

    def _render(self, mode='human', close=False):
        print('Car at(axis0, axis1) {:2},{:2} with velocity(left,up, right) {:2},{:2},{:2}'
              .format(self.axis0, self.axis1,
                      self.velocity_left, self.velocity_up, self.velocity_right))
        road_with_car = self.road.copy()
        road_with_car[(self.axis0, self.axis1)] = 8
        print(road_with_car)

    def _close(self):
        self.road = None

    def _seed(self, seed=None):
        return super()._seed(seed)

    def _reset(self):
        self.road = self.road_template
        starts = np.argwhere(self.road == CellType.START.value)
        start = starts[np.random.choice(starts.shape[0])]
        self.axis0 = start[0]
        self.axis1 = start[1]
        self.velocity_left = 0
        self.velocity_up = 0
        self.velocity_right = 0

    def _observations(self):
        return (self.axis0, self.axis1,
                self.velocity_left, self.velocity_up, self.velocity_right)

    def _step(self, action):
        action = Action(action)
        log.debug('Action chosen: {}'.format(action))
        if action.is_left():
            self.velocity_left = self.add_velocity(self.velocity_left + action.velocity())
        elif action.is_up():
            self.velocity_up = self.add_velocity(self.velocity_up + action.velocity())
        elif action.is_right():
            self.velocity_right = self.add_velocity(self.velocity_right + action.velocity())

        side = self.velocity_right - self.velocity_left
        self.axis1 += side
        self.axis0 -= self.velocity_up
        pos = (self.axis0, self.axis1)
        reward, done = -1, False
        if self.is_out_of_range(pos) or self.road[pos] == CellType.OFF.value:
            reward = -5
            done = True
        elif self.road[pos] == CellType.STOP:
            reward = 1
            done = True

        return self._observations(), reward, done, self.road


if __name__ == '__main__':
    env = RaceCarEnv()
    for i in range(1):
        print('Episode: {}'.format(i))
        env.reset()
        done = False
        while not done:
            env.render()
            obs, reward, done, road = env.step(env.action_space.sample())
            log.debug('Reward(done={}): {}'.format(done, reward))
