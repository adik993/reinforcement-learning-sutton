from envs.RandomWalkEnv import RandomWalk
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tools

TRUE_VALUES = np.array([0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1])
ALPHA = 0.1


class State:
    def __init__(self, position, next_position, action, reward):
        self.position = position
        self.next_position = next_position
        self.action = action
        self.reward = reward

    def __str__(self) -> str:
        return 'State(position={}, action={}, reward={})' \
            .format(self.position, self.action, self.reward)

    def __repr__(self) -> str:
        return self.__str__()


def generate_episode(env: RandomWalk):
    history = []
    done = False
    obs = env.reset()
    while not done:
        prev_obs = obs
        action = env.action_space.sample()
        obs, reward, done, aux = env.step(action)
        state = State(prev_obs, obs, action, reward)
        history.append(state)
    return history


def monte_carlo(value, history, alpha=ALPHA):
    value = value.copy()
    for i, state in enumerate(history):
        ret = np.mean([state.reward for state in history[i:]])
        value[state.position] += alpha * (ret - value[state.position])
    return value


def td0(value, history, alpha=ALPHA, gamma=1):
    value = value.copy()
    for i, state in enumerate(history):
        value[state.position] += alpha * (state.reward + gamma * value[state.next_position] - value[state.position])
    return value


def get_initial_value(n_states):
    values = np.full((n_states,), 0.5)
    values[0], values[-1] = 0, 0
    return values


def rmse(predicted, actual):
    return np.sqrt(np.mean((predicted - actual) ** 2))


def perform_alpha_sim(alphas: dict, algorithm, n_states: int, episodes=100, n_average=100):
    for alpha in alphas.keys():
        curr = np.zeros((episodes + 1,))
        for i in range(n_average):
            value = get_initial_value(n_states)
            curr[0] += rmse(value[1:-1], TRUE_VALUES[1:-1])
            for ep in range(episodes):
                history = generate_episode(env)
                value = algorithm(value, history, alpha=alpha)
                curr[ep + 1] += rmse(value[1:-1], TRUE_VALUES[1:-1])
        alphas[alpha] = curr / n_average


if __name__ == '__main__':
    env = RandomWalk()
    n_states = env.observation_space.n
    mc_value = get_initial_value(n_states)
    td0_value = get_initial_value(n_states)
    checkpoints = [0, 1, 10, 100]
    td_value_checkpoints = {}
    mc_value_checkpoints = {}
    if 0 in checkpoints:
        td_value_checkpoints[0] = td0_value.copy()
        mc_value_checkpoints[0] = mc_value.copy()
    for ep in range(100):
        history = generate_episode(env)
        mc_value = monte_carlo(mc_value, history)
        td0_value = td0(td0_value, history)
        if ep + 1 in checkpoints:
            td_value_checkpoints[ep + 1] = td0_value.copy()
            mc_value_checkpoints[ep + 1] = mc_value.copy()

    mc_alphas = {0.01: [], 0.02: [], 0.03: [], 0.04: []}
    td0_alphas = {0.05: [], 0.1: [], 0.15: []}
    perform_alpha_sim(mc_alphas, monte_carlo, n_states)
    perform_alpha_sim(td0_alphas, td0, n_states)

    x = np.arange(n_states - 2)
    true_trace = go.Scatter(x=n_states, y=TRUE_VALUES[1:-1], name='True Values')
    mc_value_traces = []
    td0_value_traces = []
    for n in checkpoints:
        td0_value_traces.append(go.Scatter(x=n_states, y=td_value_checkpoints[n][1:-1], name='TD(0) - {}'.format(n)))
        mc_value_traces.append(
            go.Scatter(x=n_states, y=mc_value_checkpoints[n][1:-1], name='Monte Carlo - {}'.format(n)))
    fig = go.Figure(data=[true_trace] + mc_value_traces + td0_value_traces)
    fig = tools.make_subplots(2, 2, subplot_titles=('Monte Carlo - Estimated Values', 'TD(0) - Estimated Values',
                                                    'Monte Carlo - RMSE', 'TD(0) - RMSE'))

    mc_rmse_traces = []
    td0_rmse_traces = []
    for alpha, data in mc_alphas.items():
        mc_rmse_traces.append(go.Scatter(y=data, name='Monte Carlo - {}'.format(alpha)))
    for alpha, data in td0_alphas.items():
        td0_rmse_traces.append(go.Scatter(y=data, name='TD(0) - {}'.format(alpha)))

    fig.append_trace(true_trace, 1, 1)
    fig.append_trace(true_trace, 1, 2)
    for trace in mc_value_traces:
        fig.append_trace(trace, 1, 1)
    for trace in td0_value_traces:
        fig.append_trace(trace, 1, 2)
    for trace in mc_rmse_traces:
        fig.append_trace(trace, 2, 1)
    for trace in td0_rmse_traces:
        fig.append_trace(trace, 2, 2)

    fig['layout']['title'] = 'Random Walk'
    fig['layout']['xaxis1']['title'] = 'State'
    fig['layout']['xaxis2']['title'] = 'State'
    fig['layout']['yaxis1']['title'] = 'Estimated Value'
    fig['layout']['yaxis2']['title'] = 'Estimated Value'

    fig['layout']['xaxis3']['title'] = 'Walks / Episodes'
    fig['layout']['xaxis4']['title'] = 'Walks / Episodes'
    fig['layout']['yaxis3']['title'] = 'RMSE averaged over states'
    fig['layout']['yaxis4']['title'] = 'RMSE averaged over states'
    py.plot(fig)
