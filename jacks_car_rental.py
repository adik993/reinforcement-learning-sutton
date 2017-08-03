import numpy as np
from scipy.misc import factorial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

RQ1 = 3
RT1 = 3

RQ2 = 4
RT2 = 2

MAX_MOVE = 5
MAX_CARS = 20

MOVE_COST = 2
RENTAL_EARN = 10

DISCOUNT = 0.9

# An up bound for poisson distribution
# If n is greater than this value, then the probability of getting n is truncated to 0
POISSON_UP_BOUND = 11

# Probability for poisson distribution
# @lam: lambda should be less than 10 for this function
poissonBackup = dict()


def poisson_prob(n, lam):
    global poissonBackup
    key = n * 10 + lam
    if key not in poissonBackup.keys():
        poissonBackup[key] = np.exp(-lam) * lam ** n / factorial(n)
    return poissonBackup[key]


# Positive action means move from R1 to R2 negative from R2 to R1

def next_state(state, action):
    n_cars1 = state[0]
    n_cars2 = state[1]
    n_move = np.abs(action)
    if action > 0:
        n_cars1 -= n_move
        n_cars2 += n_move
    if action < 0:
        n_cars1 += n_move
        n_cars2 -= n_move
    n_cars1 = min(n_cars1, MAX_CARS)
    n_cars1 = max(n_cars1, 0)
    n_cars2 = min(n_cars2, MAX_CARS)
    n_cars2 = max(n_cars2, 0)
    return int(n_cars1), int(n_cars2)


def all_next_states(state):
    n_cars1 = state[0]
    n_cars2 = state[1]
    states = []
    for i in range(0, min(n_cars1 + 1, MAX_MOVE + 1)):
        states.append((next_state(state, i), i))
    for i in range(0, min(n_cars2 + 1, MAX_MOVE + 1)):
        states.append((next_state(state, -i), -i))
    return states


def v(value, state, action):
    # rq_r1 = poisson(RQ1)
    # rq_r2 = poisson(RQ2)
    # rt_r1 = poisson(RT1)
    # rt_r2 = poisson(RT2)
    rt_r1 = RT1
    rt_r2 = RT2
    returns = 0.0

    # Move cars
    returns -= abs(action) * MOVE_COST
    next = next_state(state, action)

    for rq_r1 in range(0, POISSON_UP_BOUND):
        for rq_r2 in range(0, POISSON_UP_BOUND):
            # Reset number of cars
            n_cars1 = next[0]
            n_cars2 = next[1]

            # Rent cars
            n_rent_cars1 = min(n_cars1, rq_r1)
            n_rent_cars2 = min(n_cars2, rq_r2)
            prob = poisson_prob(rq_r1, RQ1) * poisson_prob(rq_r2, RQ2)
            reward = (n_rent_cars1 + n_rent_cars2) * RENTAL_EARN
            n_cars1 -= n_rent_cars1
            n_cars2 -= n_rent_cars2

            # Returned cars
            n_cars1 = int(min(MAX_CARS, n_cars1 + rt_r1))
            n_cars2 = int(min(MAX_CARS, n_cars2 + rt_r2))

            returns += prob * (reward + DISCOUNT * value[n_cars1, n_cars2])

    return returns


def policy_eval(value, policy):
    next_values = value.copy()
    for n_cars1 in range(value.shape[0]):
        for n_cars2 in range(value.shape[1]):
            # Evaluate state
            val = v(value, (n_cars1, n_cars2), policy[n_cars1, n_cars2])
            # Update values
            next_values[n_cars1, n_cars2] = val
    return next_values


def policy_improvement(value, policy):
    next_policy = np.zeros(policy.shape)
    for n_cars1 in range(value.shape[0]):
        for n_cars2 in range(value.shape[1]):
            states = all_next_states((n_cars1, n_cars2))
            i = np.argmax([v(value, (n_cars1, n_cars2), action) for state, action in states]).flatten()[0]
            next_policy[n_cars1, n_cars2] = states[i][1]
    return next_policy


if __name__ == '__main__':
    policy = np.zeros((MAX_CARS + 1, MAX_CARS + 1))
    value = np.zeros((MAX_CARS + 1, MAX_CARS + 1))

    while True:
        prev_value = value
        value = policy_eval(value, policy)
        print('Value cahnge:', np.abs(prev_value - value).sum())

        if np.abs(prev_value - value).sum() < 1e-4:
            prev_policy = policy.copy()
            policy = policy_improvement(value, policy)
            policy_changes = (policy != prev_policy).sum()
            print('Policy change:', policy_changes)
            if policy_changes == 0:
                break

    print(value)
    print(policy)
    X, Y = np.meshgrid(np.arange(0, value.shape[0]), np.arange(0, value.shape[1]))
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')
    # ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, value, cmap='jet')
    fig.colorbar(surf)
    sub = fig.add_subplot(122)
    cax = sub.matshow(policy)
    fig.colorbar(cax)
    plt.tight_layout()
    plt.show()
