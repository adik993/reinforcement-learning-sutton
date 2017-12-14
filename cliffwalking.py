from gym import Env

from envs.CliffWalkingEnv import CliffWalking
from log import make_logger
from windy_gridworld import Sarsa, generate_episode

log = make_logger(__name__)


class QLearning(Sarsa):
    def __init__(self, env: Env, alpha=0.5, gamma=1, epsilon=0.1):
        super().__init__(env, alpha, gamma, epsilon)

    def greedy_value(self, state):
        return self.action_value[state].max()

    def on_new_state(self, prev_state, action, reward, next_state, done):
        q = self.action_value[prev_state][action]
        q_next = self.greedy_value(next_state)
        self.action_value[prev_state][action] += self.alpha * (reward + self.gamma * q_next - q)


if __name__ == '__main__':
    env = CliffWalking()
    algorithm = QLearning(env, alpha=0.5, gamma=1, epsilon=0.1)
    for ep in range(int(1e2)):  # 1e4 for Sarsa
        moves = generate_episode(env, algorithm)
        log.info('Episode no. {} done in moves {}'.format(ep, moves))

    log.info('Done learning!')
    algorithm.epsilon = 0
    moves = generate_episode(env, algorithm, render=True)
    env.close()
    log.info('Learned to complete in moves: {}'.format(moves))
