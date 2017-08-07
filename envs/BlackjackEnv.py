from gym import Env
import numpy as np
from gym.spaces.discrete import Discrete
from gym.spaces.tuple_space import Tuple

ACE_CARD = 1
ACE_VALUE = 11
BLACKJACK = 21

PLAYER_MIN = 12

ACTION_STICK = 0
ACTION_HIT = 1
ACTIONS = [ACTION_STICK, ACTION_HIT]
N_ACTIONS = len(ACTIONS)

DEALER_SICK_SUM = 17


class Blackjack(Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.seed_num = None
        self.dealer = []
        self.player = []
        # ACE, 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King
        self.deck = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10])
        self.action_space = Discrete(N_ACTIONS)
        self.observation_space = Tuple((Discrete(11), Discrete(32), Discrete(2)))
        self.reward_range = (-1, 1)
        self.dealer_stop = DEALER_SICK_SUM

    def _render(self, mode='human', close=False):
        print('Dealer: sum={:2} cards={:4}'.format(self.calculate_hand_sum(self.dealer), str(self.dealer)), end=' ')
        print('Player: sum={:2} cards={}'.format(self.calculate_hand_sum(self.player), str(self.player)))

    def _step(self, action):
        assert self.action_space.contains(action)
        done = False
        if action == ACTION_HIT:
            self.player += self.draw_card()
            if self.is_busted(self.player):
                done = True
        else:
            done = True
            while self.calculate_hand_sum(self.dealer) < self.dealer_stop:
                self.dealer += self.draw_card()

        if done:
            reward = self.calculate_reward()
        else:
            reward = 0
        return self._observation(), reward, done, self._auxiliary()

    def _reset(self):
        self.player = list(self.draw_card(2))
        while self.calculate_hand_sum(self.player) < PLAYER_MIN:
            self.player += self.draw_card(1)
        self.dealer = self.draw_card()
        return self._observation()

    def _seed(self, seed=None):
        self.seed_num = seed
        return [self.seed_num]

    def draw_card(self, n=1):
        return list(np.random.choice(self.deck, n))

    def calculate_hand_sum(self, cards):
        if self.has_usable_ace(cards):
            return sum(cards) + 10
        else:
            return sum(cards)

    def has_usable_ace(self, player):
        return ACE_CARD in player and sum(player) + 10 <= BLACKJACK

    def is_busted(self, player):
        return self.calculate_hand_sum(player) > BLACKJACK

    def calculate_reward(self):
        if self.is_busted(self.player):
            return -1
        elif self.is_busted(self.dealer):
            return 1
        elif self.is_natural(self.player):
            return 0 if self.is_natural(self.dealer) else 1
        elif self.calculate_hand_sum(self.player) == self.calculate_hand_sum(self.dealer):
            return 0
        else:
            return 1 if self.calculate_hand_sum(self.player) > self.calculate_hand_sum(self.dealer) else -1

    def is_natural(self, player):
        return self.calculate_hand_sum(player) == BLACKJACK and len(player) == 2

    def _observation(self):
        return self.calculate_hand_sum(self.dealer), \
               self.calculate_hand_sum(self.player), \
               self.has_usable_ace(self.player)

    def _auxiliary(self):
        return BlackjackAuxiliary(self.dealer, self.player)


class BlackjackAuxiliary:
    def __init__(self, dealer, player):
        self.player_cards = player
        self.dealer_cards = dealer


def policy(observation):
    if observation[1] < 20:
        return ACTION_HIT
    else:
        return ACTION_STICK


if __name__ == '__main__':
    env = Blackjack()

    for episode in range(10):
        print('Episode no: {}'.format(episode))
        done = False
        observation = env.reset()
        while not done:
            env.render()
            observation, reward, done, auxiliary = env.step(policy(observation))
            if done:
                env.render()
            print('Reward: {}'.format(reward))
