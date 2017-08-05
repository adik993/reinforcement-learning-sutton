import numpy as np
import logging

logging.setLogRecordFactory(logging.LogRecord)
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)-15s - %(levelname)-5s - %(message)s')

NO_ACE_LAYER = 0
ACE_LAYER = 1
N_USABLE_ACE_LAYERS = 2

DEALER_MIN = 1  # ACE is 1 or 10
DEALER_MAX = 10  # Max in one card
N_DEALER_CARD_SUM_POSSIBILITIES = DEALER_MAX - DEALER_MIN + 1

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

REWARD_WIN = 1
REWARD_DRAW = 0
REWARD_LOSS = -1

BLACKJACK = 21

GAME_STATE_IN_PROGRESS = 0
GAME_STATE_WIN = 1
GAME_STATE_LOSE = 2
GAME_STATE_DRAW = 3


class State:
    def __init__(self, dealer, player):
        self.dealer = list(dealer)
        self.player = list(player)
        self.dealer_sum = calculate_hand_sum(self.dealer)
        self.player_sum = calculate_hand_sum(self.player)
        self.player_has_usable_ace = has_ace_usable(self.player)

    def get_policy_player_sum(self):
        return self.player_sum - PLAYER_MIN - 1

    def get_policy_dealer_sum(self):
        return self.dealer_sum - DEALER_MIN - 1

    def get_policy_has_usable_ace(self):
        return ACE_LAYER if self.player_has_usable_ace else NO_ACE_LAYER

    def __str__(self):
        return 'State(dealer_sum={:2} dealer_cards={} player_sum({})={:2}) player_cards={}'.format(
            self.dealer_sum, self.dealer,
            'has ace' if self.player_has_usable_ace else 'no  ace',
            self.player_sum, self.player)

    def __repr__(self):
        return self.__str__()


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


def determine_game_state(state):
    dealer = state.dealer_sum
    player = state.player_sum
    if player > BLACKJACK:
        return GAME_STATE_LOSE
    elif dealer > BLACKJACK:
        return GAME_STATE_WIN
    elif dealer == player:
        return GAME_STATE_DRAW
    elif player > dealer:
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


def log_card(who, card):
    value = card_value(card)
    if card == ACE_CARD:
        logging.debug('{} drawn: {:2} of value {}'.format(who, card, value))
    else:
        logging.debug('{} drawn: {:2} of value {:2}'.format(who, card, value))


def generate_episode(player_policy, dealer_policy):
    logging.debug('Generating episodes')
    history = []
    dealer_hidden = draw_card()
    dealer = [draw_card()]
    player = list(draw_card(2))
    state = State(dealer, player)
    history.append(state)
    logging.debug('Initial state: {}'.format(state))
    if calculate_hand_sum(player) >= BLACKJACK:
        logging.debug('Player has blackjack from initial hand: {}'.format(state))

    # Player plays seeing only one dealers card
    logging.debug('Player let\'s play')
    action = ACTION_HIT
    while state.player_sum < BLACKJACK and action == ACTION_HIT:
        # Below PLAYER_MIN its boring above start using policy
        action = ACTION_HIT if state.player_sum < PLAYER_MIN else player_policy[
            state.get_policy_dealer_sum(), state.get_policy_player_sum(), int(state.get_policy_has_usable_ace())]
        if action == ACTION_HIT:
            card = draw_card()
            log_card('Player', card)
            player.append(card)
            state = State(dealer, player)
            # If things got interesting start remembering states
            if state.player_sum >= PLAYER_MIN:
                history.append(State(dealer, player))

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
    action_value = np.zeros((N_DEALER_CARD_SUM_POSSIBILITIES, N_PLAYER_CARDS_SUM_POSSIBILITIES, N_USABLE_ACE_LAYERS))
    player_policy = np.ones(action_value.shape)
    dealer_policy = np.ones((BLACKJACK,))
    for i in range(100):
        logging.info('Episode: {}'.format(generate_episode(player_policy, dealer_policy)))
