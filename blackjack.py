import logging
from collections import defaultdict

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from envs.BlackjackEnv import Blackjack, ACE_VALUE
from log import make_logger

NO_ACE_LAYER = 0
ACE_LAYER = 1
N_USABLE_ACE_LAYERS = 2

DEALER_MIN = 1  # ACE is 1 or 10
DEALER_MAX = 10  # Max in one card
N_DEALER_CARD_SUM_POSSIBILITIES = DEALER_MAX - DEALER_MIN + 1

PLAYER_INIT_STICK_SUM = 20
PLAYER_MIN = 12  # Below 12 always hit
PLAYER_MAX = 21  # Blackjack :)
N_PLAYER_CARDS_SUM_POSSIBILITIES = PLAYER_MAX - PLAYER_MIN + 1


class State:
    def __init__(self, dealer_sum, player_sum, player_has_usable_ace):
        self.dealer_sum = dealer_sum
        self.player_sum = player_sum
        self.player_has_usable_ace = player_has_usable_ace

    def get_policy_player_sum(self):
        return self.player_sum - PLAYER_MIN

    def get_policy_dealer_sum(self):
        if self.dealer_sum == ACE_VALUE:
            return 0
        else:
            return self.dealer_sum - DEALER_MIN

    def get_policy_has_usable_ace(self):
        return ACE_LAYER if self.player_has_usable_ace else NO_ACE_LAYER

    def __str__(self):
        return 'State(dealer_sum={:2} player_sum({})={:2})'.format(
            self.dealer_sum,
            'has ace' if self.player_has_usable_ace else 'no  ace',
            self.player_sum)

    def __repr__(self):
        return self.__str__()

    def to_policy_key(self):
        ace_layer = ACE_LAYER if self.player_has_usable_ace else NO_ACE_LAYER
        return self.get_policy_dealer_sum(), self.get_policy_player_sum(), ace_layer


def generate_episode(env: Blackjack, player_policy, ep_no):
    history = []
    done = False
    observation = env.reset()
    while not done:
        state = State(*observation)
        history.append(state)
        log.debug('Episode no {}: {}'.format(ep_no, state))
        observation, reward, done, auxiliary = env.step(player_policy[state.to_policy_key()])
    return history, reward


if __name__ == '__main__':
    log = make_logger(__name__, logging.DEBUG)
    env = Blackjack()
    state_value = np.zeros((N_DEALER_CARD_SUM_POSSIBILITIES, N_PLAYER_CARDS_SUM_POSSIBILITIES, N_USABLE_ACE_LAYERS))
    player_policy = np.ones(state_value.shape, dtype=np.int32)
    player_policy[:, (PLAYER_INIT_STICK_SUM - PLAYER_MIN):, :] = 0
    returns = defaultdict(list)
    for i in range(100000):
        episode, reward = generate_episode(env, player_policy, i)
        log.info('Episode no {} rewarded {:2}: {}'.format(i, reward, episode))
        for state in episode:
            key = state.to_policy_key()
            returns[key].append(reward)
            state_value[key] = np.mean(returns[key])

    X, Y = np.meshgrid(np.arange(0, state_value.shape[0]) + DEALER_MIN, np.arange(0, state_value.shape[1]) + PLAYER_MIN)
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    ax.set_title('No usable ace')
    ax.set_xlabel('dealer sum')
    ax.set_ylabel('player sum')
    ax.set_xticks(np.arange(0, state_value.shape[0]) + DEALER_MIN)
    ax.set_yticks(np.arange(0, state_value.shape[1]) + PLAYER_MIN)
    surf = ax.plot_surface(X, Y, state_value[:, :, NO_ACE_LAYER].T, cmap='jet')
    fig.colorbar(surf)

    ax = fig.add_subplot(122, projection='3d')
    ax.set_title('Usable ace')
    ax.set_xlabel('dealer sum')
    ax.set_ylabel('player sum')
    ax.set_xticks(np.arange(0, state_value.shape[0]) + DEALER_MIN)
    ax.set_yticks(np.arange(0, state_value.shape[1]) + PLAYER_MIN)
    surf = ax.plot_surface(X, Y, state_value[:, :, ACE_LAYER].T, cmap='jet')
    fig.colorbar(surf)
    plt.show()
