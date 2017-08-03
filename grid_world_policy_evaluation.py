import numpy as np

np.set_printoptions(1)

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTIONS = [UP, DOWN, LEFT, RIGHT]


def reward():
    return -1


def policy(state, action):
    return 0.25


def get_state(grid, state, action):
    row = state // grid.shape[0]
    col = state % grid.shape[0]
    if action == UP:
        row = max(row - 1, 0)
    elif action == DOWN:
        row = min(row + 1, grid.shape[0] - 1)
    elif action == LEFT:
        col = max(col - 1, 0)
    elif action == RIGHT:
        col = min(col + 1, grid.shape[1] - 1)

    return grid[row, col]


def get_avg(grid, state):
    return sum([policy(state, action) * (get_state(grid, state, action) + reward()) for action in ACTIONS])


def set_state(grid, state, value):
    row = state // grid.shape[0]
    col = state % grid.shape[0]
    grid[row, col] = value


if __name__ == '__main__':
    value = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0]], np.float32)
    print(value)
    for i in range(112):
        prev = value.copy()
        for state in range(1, value.size - 1):
            avg = get_avg(prev, state)
            set_state(value, state, avg)
        print(value)
