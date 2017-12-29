import gym
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
from gym import Env
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import RMSprop

from utils import Algorithm, generate_episode, EpsilonDecay, Model, create_huber_loss, NumpyRingBufferMemory


class QValueTracker:
    def __init__(self, state):
        self.state = state
        self.history = []

    def on_new_state(self, model: Model):
        predicted = model.predict(self.state).flatten()[0]
        print('Q-Value:', predicted)
        self.history.append(predicted)


class NNModel(Model):
    def __init__(self, state_size, action_size, alpha, huber_loss_delta):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.huber_loss_delta = huber_loss_delta
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.summary()
        model.compile(loss=create_huber_loss(self.huber_loss_delta), optimizer=RMSprop(self.alpha))
        return model

    def predict(self, state, target=False):
        if target:
            return self.target_model.predict(state)
        else:
            return self.model.predict(state)

    def train(self, state, target):
        self.model.fit(state, target, batch_size=len(state), epochs=1, verbose=0)

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


class DeepQLearning(Algorithm):
    def __init__(self, env: Env, state_size, action_size, alpha=0.00025, gamma=0.99,
                 epsilon_decay=EpsilonDecay(1.0, 0.01, 0.001), n_replays=64, train_start=100000,
                 model_update_interval=1000, huber_loss_delta=1.0, memory_size=100000,
                 q_value_tracker: QValueTracker = None):
        self.env = env
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.n_replays = n_replays
        self.train_start = train_start
        self.model_update_interval = model_update_interval
        self.q_value_tracker = q_value_tracker
        self.actions = np.arange(action_size)
        self.model = NNModel(state_size, action_size, alpha, huber_loss_delta)
        self.memory = NumpyRingBufferMemory(memory_size, state_size)
        self.step = 0

    def _should_explore(self):
        return np.random.rand() <= self.epsilon_decay.value()

    def action(self, state):
        if self._should_explore():
            return self.env.action_space.sample()
        else:
            return self._greedy_action(state)

    def _greedy_action(self, state):
        return np.argmax(self.model.predict(self._prep_state(state))[0])

    def _prep_state(self, state):
        return state.reshape((-1, self.state_size))

    def _store_visit(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

    def _replay(self):
        batch = self.memory.sample(self.n_replays)
        states = batch['state']
        actions = batch['action']
        rewards = batch['reward']
        next_states = batch['next_state']
        dones = batch['done']
        q_nexts = np.max(self.model.predict(next_states, target=True), axis=1)
        q_nexts[dones] = 0
        state_targets = self.model.predict(states)
        state_targets[np.arange(len(batch)), actions] = rewards + self.gamma * q_nexts
        self.model.train(states, state_targets)

    def on_new_state(self, state, action, reward, next_state, done):
        self.step += 1
        if self.step % self.model_update_interval == 0:
            print('[Step: {:6}] Updating target model...'.format(self.step))
            self.model.update_target_model()
        state, next_state = self._prep_state(state), self._prep_state(next_state)
        self._store_visit(state, action, reward, next_state, done)
        if len(self.memory) >= self.train_start:
            self._replay()
            self.epsilon_decay.step()

        if self.q_value_tracker is not None and self.step % 100 == 0:
            self.q_value_tracker.on_new_state(self.model)


class DoubleDeepQLearning(DeepQLearning):
    def _replay(self):
        batch = self.memory.sample(self.n_replays)
        axis0 = np.arange(len(batch))
        states = batch['state']
        actions = batch['action']
        rewards = batch['reward']
        next_states = batch['next_state']
        dones = batch['done']
        next_actions = np.argmax(self.model.predict(next_states), axis=1)
        q_nexts = self.model.predict(next_states, target=True)[axis0, next_actions]
        q_nexts[dones] = 0
        state_targets = self.model.predict(states)
        state_targets[axis0, actions] = rewards + self.gamma * q_nexts
        self.model.train(states, state_targets)


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    tracker = QValueTracker(np.array([[-0.01335408, -0.04600273, -0.00677248, 0.01517507]]))
    algo = DoubleDeepQLearning(env, env.observation_space.shape[0], env.action_space.n, q_value_tracker=tracker)
    rewards = []
    for ep in range(7000):
        steps = generate_episode(env, algo, render=False)
        rewards.append(steps)
        print('[Learning] Ep: {:3}, steps: {:4}, epsilon: {:.2}'.format(ep, steps, algo.epsilon_decay.value()))

    py.plot([go.Scatter(y=tracker.history, name='Q-Value'),
             go.Scatter(y=rewards, name='Rewards')])

    # algo.epsilon_decay = EpsilonDecay(0.0, 0.0, 0.0)
    # for ep in range(100):
    #     steps = generate_episode(env, algo, render=True)
    #     print('[Evaluating] Ep: {:3}, steps: {:4}, epsilon: {:.2}'.format(ep, steps, algo.epsilon_decay.value()))
