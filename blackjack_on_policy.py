import logging
from collections import defaultdict
import plotly.offline as py
from plotly import tools
import plotly.graph_objs as go

import numpy as np

from blackjack import State, NO_ACE_LAYER, ACE_LAYER, N_DEALER_CARD_SUM_POSSIBILITIES, \
    N_PLAYER_CARDS_SUM_POSSIBILITIES, \
    DEALER_MIN, PLAYER_MIN, PLAYER_INIT_STICK_SUM, N_USABLE_ACE_LAYERS
from envs.BlackjackEnv import Blackjack, ACE_VALUE, ACTIONS, BLACKJACK, N_ACTIONS
from blackjack_action_value import ActionState
from log import make_logger

EPSILON = 0.15


def generate_episode(env: Blackjack, player_policy, ep_no):
    history = []
    done = False
    observation = env.reset()
    while not done:
        state = State(*observation)
        action = player_policy[state.to_policy_key()]
        state = ActionState(*observation, action)
        history.append(state)
        log.debug('Episode no {}: {}'.format(ep_no, state))
        observation, reward, done, auxiliary = env.step(action)
    return history, reward


def policy_improvement(episodes, player_policy, action_values):
    new_policy = player_policy.copy()
    for state in episodes:
        i = action_values[state.to_policy_key()].argmax()
        if np.random.rand() < EPSILON / N_ACTIONS:
            action = ACTIONS[np.abs(i - 1)]  # Fancy way of changing 0 to 1 or the other way
        else:
            action = ACTIONS[i]
        new_policy[state.to_policy_key()] = action
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
        episode, reward = generate_episode(env, player_policy, i)
        log.info('Episode no {} rewarded {:2}: {}'.format(i, reward, episode))
        for state in episode:
            key = state.to_state_action_key()
            returns[key].append(reward)
            action_values[key] = np.mean(returns[key])

        new_policy = policy_improvement(episode, player_policy, action_values)
        log.debug('Changes made to policy: {}'.format((new_policy != player_policy).sum()))
        player_policy = new_policy

    state_values = to_state_value(action_values)
    player_policy = to_policy(action_values)
    x = np.arange(0, state_values.shape[0]) + DEALER_MIN
    y = np.arange(0, state_values.shape[1]) + PLAYER_MIN
    label_dealer_sum = {'title': 'dealer sum', 'dtick': 1}
    label_player_sum = {'title': 'player sum', 'dtick': 1}
    kwargs = dict(x=x, y=y, hoverinfo="x+y+z")
    layout_surf = {'xaxis': label_dealer_sum, 'yaxis': label_player_sum, 'aspectratio': {'x': 1, 'y': 1, 'z': 0.5}}
    single_contour_config = {'show': True, 'highlight': False, 'project': {'z': True}}
    contours = {'x': single_contour_config, 'y': single_contour_config, 'z': {'highlight': False}}

    surface_no_ace = go.Surface(z=state_values[:, :, NO_ACE_LAYER].T, contours=contours, **kwargs)
    surface_ace = go.Surface(z=state_values[:, :, ACE_LAYER].T, contours=contours, **kwargs)
    heatmap_policy_no_ace = go.Heatmap(z=player_policy[:, :, NO_ACE_LAYER].T, **kwargs)
    heatmap_policy_ace = go.Heatmap(z=player_policy[:, :, ACE_LAYER].T, **kwargs)

    fig = tools.make_subplots(rows=2, cols=2, shared_xaxes=True,
                              specs=[[{'is_3d': True}, {'is_3d': True}],
                                     [{'is_3d': False}, {'is_3d': False}]],
                              subplot_titles=('No usable ace', 'Usable ace',
                                              'No usable ace', 'Usable ace'))
    fig.append_trace(surface_no_ace, 1, 1)
    fig.append_trace(surface_ace, 1, 2)
    fig.append_trace(heatmap_policy_no_ace, 2, 1)
    fig.append_trace(heatmap_policy_ace, 2, 2)

    fig['layout']['scene1'].update(layout_surf)
    fig['layout']['scene2'].update(layout_surf)
    fig['layout']['xaxis1'].update(label_dealer_sum)
    fig['layout']['yaxis1'].update(label_player_sum)
    fig['layout']['xaxis2'].update(label_dealer_sum)
    fig['layout']['yaxis2'].update(label_player_sum)
    py.plot(fig)
