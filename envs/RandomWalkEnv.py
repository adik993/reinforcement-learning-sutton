from gym import Env
from gym.spaces import Discrete
import numpy as np


class RandomWalk(Env):
    metadata = {'render.modes': ['human']}
    ACTION_LEFT = 0
    ACTION_RIGHT = 1

    def __init__(self):
        self.states = np.array([0, 0, 0, 0, 0, 0, 1])
        self.action_space = Discrete(2)
        self.observation_space = Discrete(7)
        self._reset()

    def _render(self, mode='human', close=False):
        tmp = self.states.copy()
        tmp[self.position] = 8
        print('Render: {}'.format(tmp))

    def _reset(self):
        self.position = 3
        return self._observations()

    def _observations(self):
        return self.position

    def _step(self, action):
        if action == RandomWalk.ACTION_LEFT:
            self.position -= 1
        else:
            self.position += 1

        done = self.position == 0 or self.position == len(self.states) - 1
        reward = self.states[self.position]

        return self._observations(), reward, done, self.states


if __name__ == '__main__':
    env = RandomWalk()
    for ep in range(100):
        print('Episode: {}'.format(ep))
        done = False
        obs = env.reset()
        while not done:
            env.render()
            obs, reward, done, aux = env.step(env.action_space.sample())
            print('Rewarded: {}'.format(reward))
