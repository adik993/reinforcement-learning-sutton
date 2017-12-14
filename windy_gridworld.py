from gym import Env

from envs.WindyGridWorldEnv import WindyGridWorld
from log import make_logger
import numpy as np

from utils import Algorithm

log = make_logger(__name__)


class Sarsa(Algorithm):
    def __init__(self, env: Env, alpha=0.5, gamma=1, epsilon=0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        obs_space = [space.n for space in env.observation_space.spaces]
        self.action_value = np.zeros(obs_space + [env.action_space.n])
        self.actions = np.arange(env.action_space.n)

    def greedy_action(self, state):
        return self.action_value[state].argmax()

    def action(self, state):
        greedy = self.greedy_action(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice([action for action in self.actions if action != greedy])
        else:
            return greedy

    def on_new_state(self, prev_state, action, reward, next_state, done):
        if done:
            return
        next_action = self.action(next_state)
        q = self.action_value[prev_state][action]
        next_q = self.action_value[next_state][next_action]
        self.action_value[prev_state][action] += self.alpha * (reward + self.gamma * next_q - q)


def generate_episode(env: Env, algorithm: Sarsa, render=False):
    done = False
    obs = env.reset()
    count = 0
    while not done:
        if render:
            env.render()
        prev_obs = obs
        action = algorithm.action(obs)
        obs, reward, done, aux = env.step(action)
        algorithm.on_new_state(prev_obs, action, reward, obs, done)
        count += 1
    return count


if __name__ == '__main__':
    env = WindyGridWorld()
    sarsa = Sarsa(env)
    for ep in range(int(1e4)):
        moves = generate_episode(env, sarsa)
        log.info('Episode no. {} done in moves: {}'.format(ep, moves))

    log.info('Done learning')
    sarsa.epsilon = 0
    moves = generate_episode(env, sarsa, render=True)
    env.close()
    log.info('Learned to complete in moves: {}'.format(moves))
