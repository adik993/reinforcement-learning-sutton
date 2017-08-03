import numpy as np
import matplotlib.pyplot as plt

PH = 0.25
MAX_MONEY = 100


def r(state):
    return 0


def v(value, state, action):
    # No win
    new_state = state - action
    val = (1 - PH) * (r(new_state) + value[new_state])
    # Win
    new_state = min(MAX_MONEY, state + action)
    val += PH * (r(new_state) + value[new_state])
    return val


def policy_evaluation(value):
    next_value = value.copy()
    for state in range(1, len(value) - 1):
        vals = []
        for bet in range(min(state, MAX_MONEY - state) + 1):
            vals.append(v(value, state, bet))
        next_value[state] = np.amax(vals)
    return next_value


def get_policy(value):
    policy = np.zeros(value.shape)
    for state in range(1, len(value) - 1):
        best_bet = 0
        best_val = 0
        for bet in range(min(state, MAX_MONEY - state) + 1):
            val = v(value, state, bet)
            if best_val <= val:
                best_val = val
                best_bet = bet
        policy[state] = best_bet
    return policy


if __name__ == '__main__':
    value = np.zeros((MAX_MONEY + 1,))
    value[MAX_MONEY] = 1
    change = float('inf')

    fig = plt.figure()
    sub = fig.add_subplot(121)
    x = np.arange(value.shape[0])[1:-1]

    while change > 1e-10:
        prev = value
        value = policy_evaluation(value)
        sub.plot(x, value[1:-1])
        change = np.abs(value - prev).sum()
        print('Value change:', change)
    policy = get_policy(value)


    sub.plot(x, value[1:-1])
    sub = fig.add_subplot(122)
    sub.plot(x, policy[1:-1], 'ro')
    plt.tight_layout()
    plt.show()
