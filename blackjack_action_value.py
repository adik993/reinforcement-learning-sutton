import logging
from collections import defaultdict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from blackjack import State, NO_ACE_LAYER, ACE_LAYER, N_DEALER_CARD_SUM_POSSIBILITIES, \
    N_PLAYER_CARDS_SUM_POSSIBILITIES, \
    DEALER_MIN, PLAYER_MIN, PLAYER_INIT_STICK_SUM, N_USABLE_ACE_LAYERS
from envs.BlackjackEnv import Blackjack, ACE_VALUE, ACTIONS, BLACKJACK, N_ACTIONS
from log import make_logger


class ActionState(State):
    def __init__(self, dealer, player, has_ace, player_action=None):
        super().__init__(dealer, player, has_ace)
        self.player_action = player_action

    def __str__(self):
        return 'ActionState(dealer_sum={:2} player_sum({})={:2} action={})'.format(
            self.dealer_sum,
            'has ace' if self.player_has_usable_ace else 'no  ace',
            self.player_sum, self.player_action)

    def to_state_action_key(self):
        return (*self.to_policy_key(), self.player_action)


def generate_episode(env: Blackjack, player_policy, init_action, ep_no):
    history = []
    done = False
    observation = env.reset()
    while not done:
        state = State(*observation)
        # Exploring starts
        action = init_action if len(history) == 0 else player_policy[state.to_policy_key()]
        state = ActionState(*observation, action)
        history.append(state)
        log.debug('Episode no {}: {}'.format(ep_no, state))
        observation, reward, done, auxiliary = env.step(action)
    return history, reward


def policy_improvement(episodes, player_policy, action_values):
    new_policy = player_policy.copy()
    for state in episodes:
        new_policy[state.to_policy_key()] = action_values[state.to_policy_key()].argmax()
    return new_policy


def to_state_value(action_values):
    values = np.zeros(action_values.shape[:-1])
    for index, value in np.ndenumerate(action_values):
        values[index[:-1]] = action_values[index[:-1]].max()
    return values


def to_policy(action_values):
    policy = np.zeros(action_values.shape[:-1])
    for index, value in np.ndenumerate(action_values):
        policy[index[:-1]] = action_values[index[:-1]].argmax()
    return policy


if __name__ == '__main__':
    log = make_logger(__name__, logging.DEBUG)
    env = Blackjack()
    action_values = np.zeros(
        (N_DEALER_CARD_SUM_POSSIBILITIES, N_PLAYER_CARDS_SUM_POSSIBILITIES, N_USABLE_ACE_LAYERS, N_ACTIONS))
    player_policy = np.ones(action_values.shape[:-1], dtype=np.int32)
    player_policy[:, (PLAYER_INIT_STICK_SUM - PLAYER_MIN):, :] = 0
    returns = defaultdict(list)
    for i in range(500000):
        episode, reward = generate_episode(env, player_policy, np.random.choice(ACTIONS), i)
        log.info('Episode no {} rewarded {:2}: {}'.format(i, reward, episode))
        for state in episode:
            key = state.to_state_action_key()
            returns[key].append(reward)
            action_values[key] = np.mean(returns[key])

        new_policy = policy_improvement(episode, player_policy, action_values)
        log.info('Changes made to policy: {}'.format((new_policy != player_policy).sum()))
        player_policy = new_policy

    state_values = to_state_value(action_values)
    player_policy = to_policy(action_values)
    X, Y = np.meshgrid(np.arange(0, state_values.shape[0]) + DEALER_MIN,
                       np.arange(0, state_values.shape[1]) + PLAYER_MIN)
    fig = plt.figure()
    ax = fig.add_subplot(221, projection='3d')
    ax.set_title('No usable ace')
    ax.set_xlabel('dealer sum')
    ax.set_ylabel('player sum')
    ax.set_xticks(np.arange(0, state_values.shape[0]) + DEALER_MIN)
    ax.set_yticks(np.arange(0, state_values.shape[1]) + PLAYER_MIN)
    surf = ax.plot_surface(X, Y, state_values[:, :, NO_ACE_LAYER].T, cmap='jet')
    fig.colorbar(surf)

    ax = fig.add_subplot(222, projection='3d')
    ax.set_title('Usable ace')
    ax.set_xlabel('dealer sum')
    ax.set_ylabel('player sum')
    ax.set_xticks(np.arange(0, state_values.shape[0]) + DEALER_MIN)
    ax.set_yticks(np.arange(0, state_values.shape[1]) + PLAYER_MIN)
    surf = ax.plot_surface(X, Y, state_values[:, :, ACE_LAYER].T, cmap='jet')
    fig.colorbar(surf)

    ax = fig.add_subplot(223)
    ax.set_title('No usable ace')
    ax.set_xlabel('dealer sum')
    ax.set_ylabel('player sum')
    surf = ax.matshow(np.flip(player_policy[:, :, NO_ACE_LAYER].T, 1))

    ax = fig.add_subplot(224)
    ax.set_title('Usable ace')
    ax.set_xlabel('dealer sum')
    ax.set_ylabel('player sum')
    surf = ax.matshow(np.flip(player_policy[:, :, ACE_LAYER].T, 1))
    plt.show()
