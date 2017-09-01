from collections import defaultdict

from envs.RandomWalkEnv import RandomWalk
from gym import Env
import numpy as np
from randomwalk import TRUE_VALUES, rmse
import plotly.offline as py
import plotly.graph_objs as go


class RandomPolicy:
    def __init__(self, env: Env):
        self.actions = np.arange(env.action_space.n)

    def __getitem__(self, item):
        return np.random.choice(self.actions)


class SpecificPolicy:
    def __init__(self, actions):
        self.actions = actions
        self.index = -1

    def __getitem__(self, item):
        self.index = (self.index + 1) % len(self.actions)
        return self.actions[self.index]


class TD:
    def __init__(self, env: Env, policy, alpha=0.1, gamma=1, lam=0.9):
        self.alpha = alpha
        self.gamma = gamma
        self.lam = lam
        self.policy = policy
        self.values = np.zeros((env.observation_space.n,))
        self.eligibility_trace = np.zeros(self.values.shape)

    def trace(self, state):
        return self.eligibility_trace[state]

    def action(self, state):
        return self.policy[state]

    def on_new_state(self, state, reward, next_state, done):
        v = self.values[state]
        v_next = self.values[next_state]
        delta = reward + self.gamma * v_next - v
        self.eligibility_trace[state] += 1
        for s in np.argwhere(self.eligibility_trace != 0):
            s = s[0]
            if done and s == next_state:
                continue
            self.values[s] += self.alpha * delta * self.eligibility_trace[s]
            self.eligibility_trace[s] = self.gamma * self.lam * self.eligibility_trace[s]


def generate_episode(env: Env, algorithm: TD):
    done = False
    obs = env.reset()
    while not done:
        prev_obs = obs
        action = algorithm.action(prev_obs)
        obs, reward, done, aux = env.step(action)
        algorithm.on_new_state(prev_obs, reward, obs, done)


def perform_lam_test(env, lams, alphas, n_avg=1, n=10):
    ret = defaultdict(lambda: np.zeros(len(alphas)))
    for lam in lams:
        for a, alpha in enumerate(alphas):
            print('Computing lam={} alpha={}'.format(lam, alpha))
            for i in range(n_avg):
                policy = SpecificPolicy([1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1])
                algorithm = TD(env, policy, lam=lam, alpha=alpha)
                for ep in range(n):
                    generate_episode(env, algorithm)
                ret[lam][a] += rmse(algorithm.values[1:-1], TRUE_VALUES[1:-1])
            ret[lam][a] /= n_avg
    return ret


if __name__ == '__main__':
    # WORK IN PROGRESS !!!!!!!
    env = RandomWalk(left_reward=-1)
    # lams = [0, .2, .4, .6, .8, .9, .95, .975, .99, 1]
    lams = [0, .2, .4, .6, .8]
    alphas = [0, .2, .4, .6, .8, 1]
    alphas = np.arange(0, 1, 0.01)
    result = perform_lam_test(env, lams, alphas)
    print(result)
    data = []
    for lam, values in result.items():
        data.append(go.Scatter(x=alphas, y=values, name='TD({})'.format(lam)))
    layout = go.Layout(
        yaxis=dict(
            range=[0, 0.6]
        )
    )
    fig = go.Figure(data=data, layout=layout)
    py.plot(fig)
