from gym import Env
from gym.spaces import Discrete
import numpy as np


class RandomWalk(Env):
    metadata = {'render.modes': ['human']}
    ACTION_LEFT = 0
    ACTION_RIGHT = 1

    def __init__(self, n_states=7, left_reward=0, right_reward=1, start_position=3):
        self.states = np.zeros((n_states,))
        self.states[0] = left_reward
        self.states[-1] = right_reward
        self.start_position = len(self.states) // 2 if start_position is None else start_position
        self.action_space = Discrete(2)
        self.observation_space = Discrete(len(self.states))
        self._reset()

    def _render(self, mode='human', close=False):
        tmp = self.states.copy()
        tmp[self.position] = 8
        print('Render: {}'.format(tmp))

    def _reset(self):
        self.position = self.start_position
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
