from gym import Env
from gym.spaces import Tuple, Discrete
import numpy as np
from envs.GridWorldEnv import GridWorld


class Maze(GridWorld):
    WALL = 2

    def __init__(self, shape=(8, 10), start=(4, 1), end=(4, 6)):
        super().__init__(shape, start, end)

    def _init_walls(self):
        raise NotImplementedError()

    def is_wall(self, position):
        return self.world[position] == Maze.WALL

    def _move(self, move):
        axis0 = np.clip(self.position[0] + move[0], 0, self.world.shape[0] - 1)
        axis1 = np.clip(self.position[1] + move[1], 0, self.world.shape[1] - 1)
        if not self.is_wall((axis0, axis1)):
            self.position = (axis0, axis1)

    def _reset(self):
        self.world[:] = 0
        self._init_walls()
        return super()._reset()

    def _step(self, action):
        obs, reward, done, aux = super()._step(action)
        reward = 0
        if done:
            reward = 1
        return obs, reward, done, aux


class BasicMaze(Maze):
    def __init__(self):
        super().__init__(shape=(6, 9), start=(2, 0), end=(0, 8))

    def _init_walls(self):
        self.world[1:4, 2] = Maze.WALL
        self.world[4, 5] = Maze.WALL
        self.world[0:3, 7] = Maze.WALL


class MazeShortLong(Maze):
    def __init__(self):
        super().__init__(shape=(6, 9), start=(0, 8), end=(5, 3))
        self.t = 0
        self.change_time = 1000

    def _init_walls(self):
        self.world[3, 0:8] = Maze.WALL

    def _move_wall(self):
        self.world[3, 0] = 0
        self.world[3, 8] = Maze.WALL

    def _reset(self):
        self.t = 0
        return super()._reset()

    def _step(self, action):
        if self.t == self.change_time:
            self._move_wall()
        self.t += 1
        obs, reward, done, aux = super()._step(action)
        if done:
            self.position = self.start
            done = False
        return obs, reward, done, aux


class MazeLongShort(MazeShortLong):
    def __init__(self):
        super().__init__()
        self.change_time = 3000

    def _init_walls(self):
        self.world[3, 1:9] = Maze.WALL

    def _move_wall(self):
        self.world[3, 8] = 0


if __name__ == '__main__':
    env = MazeLongShort()

    obs = env.reset()
    for i in range(1001):
        env.render()
        obs, reward, done, aux = env.step(env.action_space.sample())
        print('Reward:', reward)
