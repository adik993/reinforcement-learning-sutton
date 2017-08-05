from collections import defaultdict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import numpy as np
import logging

logging.setLogRecordFactory(logging.LogRecord)
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s - %(levelname)-5s - %(message)s')

NO_ACE_LAYER = 0
ACE_LAYER = 1
N_USABLE_ACE_LAYERS = 2

DEALER_SICK_SUM = 17
DEALER_MIN = 1  # ACE is 1 or 10
DEALER_MAX = 10  # Max in one card
N_DEALER_CARD_SUM_POSSIBILITIES = DEALER_MAX - DEALER_MIN + 1

PLAYER_INIT_STICK_SUM = 20
PLAYER_MIN = 12  # Below 12 always hit
PLAYER_MAX = 21  # Blackjack :)
N_PLAYER_CARDS_SUM_POSSIBILITIES = PLAYER_MAX - PLAYER_MIN + 1

# ACE, 2, 3, 4, 5, 6, 7, 8, 9, 10, Jack, Queen, King
CARDS = np.arange(1, 13 + 1)
ACE_CARD = 1
JACK_CARD = 11
QUEEN_CARD = 12
KING_CARD = 13

ACTION_STICK = 0
ACTION_HIT = 1
ACTIONS = [ACTION_STICK, ACTION_HIT]
N_ACTIONS = len(ACTIONS)

REWARD_WIN = 1
REWARD_DRAW = 0
REWARD_LOSS = -1

BLACKJACK = 21

GAME_STATE_IN_PROGRESS = 0
GAME_STATE_WIN = 1
GAME_STATE_LOSE = 2
GAME_STATE_DRAW = 3


class State:
    def __init__(self, dealer, player, player_action=None):
        self.dealer = list(dealer)
        self.player = list(player)
        self.player_action = player_action
        self.dealer_sum = calculate_hand_sum(self.dealer)
        self.player_sum = calculate_hand_sum(self.player)
        self.player_has_usable_ace = has_ace_usable(self.player)

    def get_policy_player_sum(self):
        return self.player_sum - PLAYER_MIN

    def get_policy_dealer_sum(self):
        if self.dealer[0] == ACE_CARD:
            return DEALER_MIN - 1
        else:
            return self.dealer_sum - DEALER_MIN

    def get_policy_has_usable_ace(self):
        return ACE_LAYER if self.player_has_usable_ace else NO_ACE_LAYER

    def __str__(self):
        return 'State(dealer_sum={:2} dealer_cards={} player_sum({})={:2}) player_action={} player_cards={}'.format(
            self.dealer_sum, self.dealer,
            'has ace' if self.player_has_usable_ace else 'no  ace',
            self.player_sum, self.player_action, self.player)

    def __repr__(self):
        return self.__str__()

    def to_key(self):
        ace_layer = ACE_LAYER if self.player_has_usable_ace else NO_ACE_LAYER
        return self.get_policy_dealer_sum(), self.get_policy_player_sum(), ace_layer


def card_value(card):
    if card == ACE_CARD:
        return 1, 11
    elif card >= JACK_CARD:
        return 10
    else:
        return card


def decide_ace_value(hand):
    ace_values = card_value(ACE_CARD)
    if hand + max(ace_values) <= BLACKJACK:
        value = max(ace_values)
    else:
        value = min(ace_values)
    return value


def calculate_hand_sum(cards):
    hand = 0
    aces = 0
    for card in cards:
        if card == ACE_CARD:
            aces += 1
        else:
            hand += card_value(card)
    while aces > 0:
        hand += decide_ace_value(hand)
        aces -= 1
    return hand


def draw_card(n=1):
    if n == 1:
        return np.random.choice(CARDS)
    else:
        return np.random.choice(CARDS, n)


def has_ace_usable(cards):
    hand = 0
    aces = 0
    for card in cards:
        if card == ACE_CARD:
            aces += 1
        else:
            hand += card_value(card)
    if aces > 0:
        ace_values = card_value(ACE_CARD)
        return decide_ace_value(hand) == max(ace_values)
    else:
        return False


def has_blackjack(card_sum):
    return card_sum == BLACKJACK


def has_natural(cards, card_sum):
    return len(cards) == 2 and has_blackjack(card_sum)


def determine_game_state(state):
    dealer_sum = state.dealer_sum
    player_sum = state.player_sum
    if player_sum > BLACKJACK:
        return GAME_STATE_LOSE
    elif dealer_sum > BLACKJACK:
        return GAME_STATE_WIN
    elif dealer_sum == player_sum:
        if has_natural(state.player, player_sum) and not has_natural(state.dealer, dealer_sum):
            return GAME_STATE_WIN
        else:
            return GAME_STATE_DRAW
    elif player_sum > dealer_sum:
        return GAME_STATE_WIN
    else:
        return GAME_STATE_LOSE


def get_reward(game_state):
    if game_state == GAME_STATE_WIN:
        return 1
    elif game_state == GAME_STATE_LOSE:
        return -1
    elif game_state == GAME_STATE_DRAW:
        return 0
    else:
        raise Exception('Invalid game state {}'.format(game_state))


def is_player_busted(state):
    if state.player_sum > BLACKJACK:
        return True
    else:
        return False


def should_remember(state):
    return state.player_sum >= PLAYER_MIN


def log_card(who, card):
    value = card_value(card)
    if card == ACE_CARD:
        logging.debug('{} drawn: {:2} of value {}'.format(who, card, value))
    else:
        logging.debug('{} drawn: {:2} of value {:2}'.format(who, card, value))


def generate_episode(player_policy, dealer_policy):
    logging.debug('Generating episodes')
    history = []

    # Exploring starts
    action = np.random.choice(ACTIONS)
    dealer_hidden = draw_card()
    dealer = [draw_card()]
    player = list(draw_card(2))
    while calculate_hand_sum(player) < PLAYER_MIN or calculate_hand_sum(player) > PLAYER_MAX:
        player = list(draw_card(2))
    state = State(dealer, player, action)
    history.append(state)
    logging.debug('Initial state: {}'.format(state))
    if calculate_hand_sum(player) >= BLACKJACK:
        logging.debug('Player has blackjack from initial hand: {}'.format(state))

    # Player plays seeing only one dealers card
    logging.debug('Player let\'s play')
    while state.player_sum < BLACKJACK and action == ACTION_HIT:
        # Below PLAYER_MIN its boring above start using policy
        action = ACTION_HIT if state.player_sum < PLAYER_MIN else player_policy[state.to_key()]
        if action == ACTION_HIT:
            card = draw_card()
            log_card('Player', card)
            player.append(card)
            state = State(dealer, player, action)
            # If things got interesting start remembering states
            if should_remember(state):
                history.append(state)

    # Remove bust state
    busted = is_player_busted(history[-1])
    if busted: logging.debug('Player busted: {}'.format(history[-1]))
    if busted and len(history) > 1:
        logging.debug('Remove bust state: {}'.format(history[-1]))
        history = history[:-1]

    # Dealer shows a card and plays, it doest append history, but is needed to determine win or loss
    dealer.append(dealer_hidden)
    state = State(dealer, player)
    logging.debug('Dealer showed card: {}'.format(state))
    logging.debug('Dealer let\'s play')
    action = ACTION_HIT
    while state.dealer_sum < BLACKJACK and action == ACTION_HIT:
        action = dealer_policy[state.dealer_sum]
        if action == ACTION_HIT:
            card = draw_card()
            log_card('Dealer', card)
            dealer.append(card)
            state = State(dealer, player)

    game_state = determine_game_state(state)
    reward = get_reward(game_state)
    logging.debug('Game reward is {} for final state {}'.format(reward, state))
    return history, reward


if __name__ == '__main__':
    state_value = np.zeros((N_DEALER_CARD_SUM_POSSIBILITIES, N_PLAYER_CARDS_SUM_POSSIBILITIES, N_USABLE_ACE_LAYERS))
    player_policy = np.ones(state_value.shape)
    player_policy[:, (PLAYER_INIT_STICK_SUM - PLAYER_MIN):, :] = 0
    dealer_policy = np.ones((BLACKJACK + 1,))  # Quick solution assume dealer can have sums 0 up to 21 so 22 states
    dealer_policy[DEALER_SICK_SUM:] = 0  # Stick at DEALER_SICK_SUM or more
    returns = defaultdict(list)
    for i in range(500000):
        episode, reward = generate_episode(player_policy, dealer_policy)
        logging.info('Episode no {} rewarded {:2}: {}'.format(i, reward, episode))
        for state in episode:
            key = state.to_key()
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
