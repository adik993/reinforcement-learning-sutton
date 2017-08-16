import logging
import os
from enum import Enum
from itertools import product

import numpy as np
from gym import Env
from gym.spaces import Tuple, Discrete

from log import make_logger

log = make_logger(__name__, logging.INFO)

ACTIONS = list(product((-1, 0, 1), (-1, 0, 1)))


class CellType(Enum):
    OFF = 0
    ROAD = 1
    START = 2
    STOP = 3

    @staticmethod
    def values():
        return list(CellType)


class Reward(Enum):
    WIN = 1
    STEP = -1
    LOOSE = -5


def load_race(path):
    log.debug('Loading race file...')
    log.debug('Current dir: {}'.format(os.getcwd()))
    log.debug('Race file: {}'.format(path))
    ret = []
    with open(path) as f:
        for line in f:
            ret.append([int(x) for x in line.strip()])
    return np.array(ret, dtype=np.int32)


class RaceDiscrete(Discrete):
    def __init__(self, env, n):
        super().__init__(n)
        self.env = env

    @staticmethod
    def is_legal(vel, action):
        axis0 = vel[0] + action[0]
        axis1 = vel[1] + action[1]
        return 0 <= axis0 <= 5 and 0 <= axis1 <= 5 and (axis0 != 0 or axis1 != 0)

    @staticmethod
    def legal_actions(vel):
        legal = []
        for i, action in enumerate(ACTIONS):
            if RaceDiscrete.is_legal(vel, action):
                legal.append(i)
        return legal

    def sample(self):
        vel = self.env._observations()[2:]
        log.debug('Velocity is: {}'.format(vel))
        legal = RaceDiscrete.legal_actions(vel)
        return np.random.choice(legal)


class RaceCarEnv(Env):
    VELOCITY_MIN = 0
    VELOCITY_MAX = 5
    metadata = {'render.modes': ['human']}

    def __init__(self, racefile_path):
        self.road_template = load_race(racefile_path)
        self.action_space = RaceDiscrete(self, len(ACTIONS))
        self.observation_space = Tuple((
            Discrete(self.road_template.shape[0]),
            Discrete(self.road_template.shape[1]),
            Discrete(RaceCarEnv.VELOCITY_MAX + 1),
            Discrete(RaceCarEnv.VELOCITY_MAX + 1),
        ))
        self.reward_range = (-5, 1)
        self._reset()

    def is_out_of_range(self, pos):
        return self.is_axis_out_of_range(pos[0], 0) or self.is_axis_out_of_range(pos[1], 1)

    def has_corssed_finish_line(self, position):
        stops = np.argwhere(self.road == CellType.STOP.value)
        stops.sort()
        return stops[0][0] <= position[0] <= stops[-1][0] and position[1] >= stops[0][1]

    def is_axis_out_of_range(self, value, axis):
        return value < 0 or value >= self.road.shape[axis]

    # Could be used if velocity isn't capped at 5
    def max_velocity(self, race_size, cur_pos=0, cur_vel=0):
        if cur_pos >= race_size:
            return cur_vel - 1
        else:
            return self.max_velocity(race_size, cur_pos + cur_vel, cur_vel + 1)

    @staticmethod
    def add_velocity(velocity):
        return min(max(velocity, RaceCarEnv.VELOCITY_MIN), RaceCarEnv.VELOCITY_MAX)

    def _render(self, mode='human', close=False):
        print('Car at(axis0, axis1) {:2},{:2} with velocity(up, right) {:2},{:2}'
              .format(self.axis0, self.axis1, self.velocity_up, self.velocity_right))
        if 0 <= self.axis0 < self.road.shape[0] and 0 <= self.axis1 < self.road.shape[1]:
            road_with_car = self.road.copy()
            road_with_car[(self.axis0, self.axis1)] = 8
            print(road_with_car)
        else:
            print('Car is off the map')

    def _close(self):
        self.road = None

    def _seed(self, seed=None):
        return super()._seed(seed)

    def _reset(self):
        self.road = self.road_template
        starts = self.get_starts()
        start = starts[np.random.choice(starts.shape[0])]
        self.axis0 = start[0]
        self.axis1 = start[1]
        self.velocity_up = 1
        self.velocity_right = 0
        return self._observations()

    def get_starts(self):
        return np.argwhere(self.road == CellType.START.value)

    def _observations(self):
        return self.axis0, self.axis1, self.velocity_up, self.velocity_right

    def _step(self, action):
        action = ACTIONS[action]
        log.debug('Action chosen: {}'.format(action))
        self.velocity_up = self.add_velocity(self.velocity_up + action[0])
        self.velocity_right = self.add_velocity(self.velocity_right + action[1])

        self.axis1 += self.velocity_right
        self.axis0 -= self.velocity_up
        pos = (self.axis0, self.axis1)
        reward, done = Reward.STEP.value, False
        if self.has_corssed_finish_line(pos):
            log.info('Got to the end!')
            reward = Reward.WIN.value
            done = True
        elif self.is_out_of_range(pos) or self.road[pos] == CellType.OFF.value:
            reward = Reward.LOOSE.value
            done = True
        return self._observations(), reward, done, self.road


if __name__ == '__main__':
    env = RaceCarEnv('../maps/race1.txt')
    print(env.observation_space)
    for i in range(100):
        print('Episode: {}'.format(i))
        env.reset()
        done = False
        while not done:
            env.render()
            obs, reward, done, road = env.step(env.action_space.sample())
            log.debug('Reward(done={}): {}'.format(done, reward))
