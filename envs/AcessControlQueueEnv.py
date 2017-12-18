from gym import Env
from gym.spaces import Discrete, Tuple
import numpy as np
from gym.wrappers import TimeLimit


class AccessControlActionSpace(Discrete):
    def __init__(self, env, n):
        super().__init__(n)
        self.env = env

    def sample(self):
        if self.env.free_servers == 0:
            return AccessControlQueue.ACTION_REJECT
        else:
            return super().sample()


class AccessControlQueue(Env):
    metadata = {'render.modes': ['human']}
    PRIORITIES = np.arange(4)
    REWARDS = [1, 2, 4, 8]
    ACTION_REJECT = 0
    ACTION_ACCEPT = 1

    def __init__(self, n_servers=10, free_prob=0.06):
        self.n_servers = n_servers
        self.free_prob = free_prob
        self.action_space = AccessControlActionSpace(self, 2)
        self.observation_space = Tuple((
            Discrete(len(AccessControlQueue.PRIORITIES)),
            Discrete(self.n_servers + 1)
        ))
        self._reset()

    def _step(self, action):
        reward = 0
        if action == AccessControlQueue.ACTION_ACCEPT:
            self._try_use_server()
            reward = AccessControlQueue.REWARDS[self.current_priority]
        # Next customer
        self.current_priority = self._pop_customer()
        self._try_free_servers()
        return self._obs(), reward, False, None

    def _reset(self):
        self.free_servers = self.n_servers
        self.current_priority = self._pop_customer()
        return self._obs()

    def _render(self, mode='human', close=False):
        print('*************************')
        print('Current ({}): {}'.format(self.current_priority, AccessControlQueue.REWARDS[self.current_priority]))
        print('Free servers:', self.free_servers)

    def _pop_customer(self):
        return np.random.choice(AccessControlQueue.PRIORITIES)

    def _try_free_servers(self):
        busy = self.n_servers - self.free_servers
        self.free_servers += np.random.binomial(busy, self.free_prob)

    def _try_use_server(self):
        if self.free_servers == 0:
            raise ValueError('Cannot accept with all servers busy')
        self.free_servers -= 1

    def _obs(self):
        return self.current_priority, self.free_servers


class AccessControlQueueTimeLimit(TimeLimit):
    def __init__(self, max_episode_steps, n_servers=10, free_prob=0.06):
        super().__init__(AccessControlQueue(n_servers=n_servers, free_prob=free_prob),
                         max_episode_steps=max_episode_steps)


if __name__ == '__main__':
    env = AccessControlQueue()
    obs = env.reset()
    for i in range(100):
        env.render()
        action = env.action_space.sample()
        obs, reward, _, _ = env.step(action)
        print('Action:', 'ACCEPT' if action == AccessControlQueue.ACTION_ACCEPT else 'REJECT')
        print('Reward:', reward)
