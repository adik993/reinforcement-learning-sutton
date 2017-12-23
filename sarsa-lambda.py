from collections import defaultdict

import numpy as np
import gym
from gym import Env

from features.TileCoding import IHT
from semi_gradient_sarsa_mountain_car import ValueFunction
from utils import Algorithm, generate_episode, epsilon_probs, randargmax, TilingFunctionCreator, Averager, \
    GymEpisodeTaskFactory, AlgorithmFactory, plot_scatters_from_dict

N_TILINGS = 8


class ValueFunctionCreator(TilingFunctionCreator):
    def __init__(self, n_tilings: int, iht: IHT):
        self.n_tilings = n_tilings
        self.iht = iht

    def create(self):
        return ValueFunction(self.n_tilings, self.iht)


class SarsaLambda(Algorithm):
    def __init__(self, env: Env, creator: TilingFunctionCreator, alpha=0.5 / N_TILINGS, lam=0.92, epsilon=0.0,
                 gamma=1.0):
        self.env = env
        self.value_func_creator = creator
        self.value_function = creator.create()
        self.alpha = alpha
        self.lam = lam
        self.epsilon = epsilon
        self.gamma = gamma
        self.actions = np.arange(env.action_space.n)
        self._reset()

    def action(self, state):
        if self.next_action is None:
            return self._action(state)
        else:
            return self.next_action

    def _reset(self):
        self.e_trace = self.value_func_creator.create()
        self.next_action = None

    def _action(self, state):
        greedy = self.greedy_action(state)
        probs = epsilon_probs(greedy, self.actions, self.epsilon)
        return np.random.choice(self.actions, p=probs)

    def greedy_action(self, state):
        array = np.array([self.value_function.estimated(state, action) for action in self.actions])
        return randargmax(array)

    def on_new_state(self, state, action, reward, next_state, done):
        if not done:
            self.next_action = self._action(next_state)
        q = self.value_function.estimated(state, action)
        q_next = 0 if done else self.value_function.estimated(next_state, self.next_action)
        delta = reward + self.gamma * q_next - q
        self.e_trace[state, action] = 1
        self.value_function[:, :] += self.alpha * delta * self.e_trace[:, :]
        self.e_trace[:, :] *= self.gamma * self.lam
        if done:
            self._reset()


class TrueOnlineSarsaLambda(SarsaLambda):
    def _reset(self):
        super()._reset()
        self.q_old = 0

    def on_new_state(self, state, action, reward, next_state, done):
        # Note value_function.x(...) and e_trace.x(...) returns same values since they use the same IHT
        if not done:
            self.next_action = self._action(next_state)
        q = self.value_function.estimated(state, action)
        q_next = 0 if done else self.value_function.estimated(next_state, self.next_action)
        x = self.value_function.x(state, action)
        delta = reward + self.gamma * q_next - q
        self.e_trace[:, :] *= self.gamma * self.lam
        self.e_trace[state, action] += 1 - self.alpha * self.gamma * self.lam * self.e_trace.estimated(state, action)
        q_delta = q - self.q_old
        self.value_function[:, :] += self.alpha * (delta + q_delta) * self.e_trace[:, :] - self.alpha * q_delta * x
        self.q_old = q_next
        if done:
            self._reset()


class SarsaLambdaFactory(AlgorithmFactory):
    def __init__(self, env: Env):
        self.env = env

    def create(self, lam, alpha) -> Algorithm:
        return SarsaLambda(env, ValueFunctionCreator(N_TILINGS, IHT(4096)), lam=lam, alpha=alpha / N_TILINGS)


class TrueOnlineSarsaLambdaFactory(AlgorithmFactory):
    def __init__(self, env: Env):
        self.env = env

    def create(self, lam, alpha) -> Algorithm:
        return TrueOnlineSarsaLambda(env, ValueFunctionCreator(N_TILINGS, IHT(4096)), lam=lam, alpha=alpha / N_TILINGS)


def average_steps_per_episode(results, n_avg):
    tmp = np.mean(results, axis=1)
    return np.sum(tmp, axis=0) / n_avg


def perform_lambda_test(n_episodes, n_avg):
    averager = Averager(GymEpisodeTaskFactory(env, n_episodes, SarsaLambdaFactory(env)))
    alphas = np.arange(1, 15) / N_TILINGS  # Those are again divided by N_TILINGS in sarsa to give final alpha value
    results = defaultdict(lambda: np.zeros(len(alphas)))
    for lam in [0, .68, .84, .92, .96, .98, .99]:
        for i, alpha in np.ndenumerate(alphas):
            results[lam][i] = averager.average((lam, alpha), n_avg, merge=average_steps_per_episode)
    plot_scatters_from_dict(results, 'lambda={}', alphas)


def perform_sarsa_lambda_comparison(n_episodes, n_avg):
    alphas = np.arange(0.2, 2.2, 0.2)  # Those are divided by N_TILINGS in sarsa to give final alpha value
    lam = 0.84
    results = defaultdict(lambda: np.zeros(len(alphas)))
    averager = Averager(GymEpisodeTaskFactory(env, n_episodes, SarsaLambdaFactory(env)))
    for i, alpha in np.ndenumerate(alphas):
        results['Sarsa(Lam) with replacing'][i] = -averager.average((lam, alpha), n_avg,
                                                                    merge=average_steps_per_episode)

    averager = Averager(GymEpisodeTaskFactory(env, n_episodes, TrueOnlineSarsaLambdaFactory(env)))
    for i, alpha in np.ndenumerate(alphas):
        results['True Online Sarsa(Lam)'][i] = -averager.average((lam, alpha), n_avg, merge=average_steps_per_episode)

    plot_scatters_from_dict(results, '{}', alphas)


if __name__ == '__main__':
    env = gym.make('MountainCar-v0')
    env._max_episode_steps = int(3e3)
    # perform_lambda_test(n_episodes=50, n_avg=40)
    perform_sarsa_lambda_comparison(n_episodes=20, n_avg=100)
