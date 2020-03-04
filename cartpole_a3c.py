import sys
from multiprocessing import Process, Queue, Event
from threading import Thread, RLock
from time import clock

import gym
import keras.backend as K
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from os.path import isfile

from utils import Algorithm, generate_episode, EpsilonDecay


class InterruptableProcess(Process):
    def __init__(self):
        super().__init__()
        self.interrupted = Event()

    def interrupt(self):
        self.interrupted.set()


class InterruptableThread(Thread):
    def __init__(self, daemon):
        super().__init__(daemon=daemon)
        self.interrupted = False

    def interrupt(self):
        self.interrupted = True


class A3CModel:
    def predict(self, state):
        raise NotImplementedError('Implement me')

    def train(self, state, action, reward, next_state, done):
        raise NotImplementedError('Implement me')


class PolicyGradientModel(A3CModel):
    def __init__(self, state_size, action_size, alpha, alpha_decay, gamma, n, c_v, c_e, clip_epsilon, n_epochs,
                 load_from_file=None):
        self.state_size = state_size
        self.action_size = action_size
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma_n = gamma ** n
        self.c_v = c_v
        self.c_e = c_e
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.lock = RLock()
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)
        self.model = self._build_model(load_from_file)
        self.graph = self._build_graph()
        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()
        self.default_graph.finalize()
        self.t = 0

    def _build_model(self, load_from_file):
        input_layer = Input(batch_shape=(None, self.state_size))
        hidden = Dense(16, activation='relu')(input_layer)
        policy_output = Dense(self.action_size, activation='softmax')(hidden)
        value_output = Dense(1, activation='linear')(hidden)
        model = Model(input_layer, [policy_output, value_output])
        if load_from_file is not None:
            model.load_weights(load_from_file)
        # noinspection PyProtectedMember
        model._make_predict_function()
        return model

    def _build_graph(self):
        state = tf.placeholder(tf.float32, [None, self.state_size], name='state')
        action = tf.placeholder(tf.int32, [None, 1], name='action')
        reward = tf.placeholder(tf.float32, [None, 1], name='reward')
        next_state = tf.placeholder(tf.float32, [None, self.state_size], name='next_state')
        done = tf.placeholder(tf.float32, [None, 1], name='done')
        old_policy = tf.placeholder(tf.float32, [None, self.action_size], name='old_policy')
        policy, value = self.model(state)
        _, next_value = self.model(next_state)
        one_hot_action = tf.reshape(tf.one_hot(action, self.action_size), [-1, self.action_size], name='one_hot_action')
        advantage = reward + self.gamma_n * next_value * done - value
        policy_ratio = tf.divide(tf.reduce_sum(policy * one_hot_action, axis=1, keep_dims=True),
                                 tf.reduce_sum(old_policy * one_hot_action, axis=1, keep_dims=True) + 1e-10,
                                 name='probability_ratio')
        # policy_log = tf.log(tf.reduce_sum(policy * one_hot_action, axis=1, keep_dims=True) + 1e-10, name='policy_log')
        # old_policy_log = tf.log(tf.reduce_sum(old_policy * one_hot_action, axis=1, keep_dims=True) + 1e-10,
        #                         name='policy_log')
        # policy_ratio = tf.exp(policy_log - old_policy_log)
        loss_policy = -tf.minimum(advantage * policy_ratio,
                                  advantage * tf.clip_by_value(policy_ratio,
                                                               1 - self.clip_epsilon,
                                                               1 + self.clip_epsilon),
                                  name='policy_loss')
        loss_value = tf.square(advantage, name='value_loss')
        entropy = tf.reduce_sum(policy * tf.log(policy + 1e-10), axis=1, keep_dims=True, name='entropy')
        loss = tf.reduce_mean(loss_policy + self.c_v * loss_value + self.c_e * entropy, name='total_loss')
        optimizer = tf.train.AdamOptimizer(self.alpha)
        minimize = optimizer.minimize(loss, name='loss_minimize')
        return state, action, reward, next_state, done, old_policy, minimize

    def predict(self, state, target=False):
        with self.lock, self.default_graph.as_default():
            return self.model.predict(state)

    def train(self, state, action, reward, next_state, done):
        with self.lock, self.default_graph.as_default():
            state_ph, action_ph, reward_ph, next_state_ph, done_ph, old_policy_ph, minimize = self.graph
            old_policy, _ = self.predict(state)
            for i in range(self.n_epochs):
                self.session.run(minimize, feed_dict={
                    state_ph: state,
                    action_ph: action,
                    reward_ph: reward,
                    next_state_ph: next_state,
                    done_ph: done,
                    old_policy_ph: old_policy
                })
            self.t += 1
            if self.t % 100 == 0:
                _, v = self.predict(np.array([[-0.01335408, -0.04600273, -0.00677248, 0.01517507]]))
                print('Value:', v.flatten()[0])


class ModelProxy(A3CModel):
    def __init__(self, uid, predict_queue, train_queue, response_queue):
        super().__init__()
        self.uid = uid
        self.predict_queue = predict_queue
        self.train_queue = train_queue
        self.response_queue = response_queue

    def predict(self, state, target=False):
        self.predict_queue.put((self.uid, state))
        return self.response_queue.get()

    def train(self, state, action, reward, next_state, done):
        self.train_queue.put((state, action, reward, next_state, done))


class Agent(Algorithm):
    def __init__(self, state_size, action_size, model: A3CModel, n, gamma, epsilon_decay: EpsilonDecay, train=True):
        super().__init__()
        self.model = model
        self.state_size = state_size
        self.action_size = action_size
        self.n = n
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.train = train
        self.actions = np.arange(self.action_size)
        self.T = 0
        self._reset()

    def _reset(self):
        self.t = 0
        self.T = sys.maxsize
        self.memory = np.zeros(self.n + 1, dtype=[('state', np.float, self.state_size),
                                                  ('action', np.int),
                                                  ('reward', np.float)])

    def action(self, state):
        if np.random.rand() <= self.epsilon_decay.value():
            return np.random.choice(self.actions)
        else:
            p, v = self.model.predict(state.reshape((1, -1)))
            return np.random.choice(self.actions, p=p.flatten())

    def _n_step_return(self, update_time):
        return sum([pow(self.gamma, t - update_time - 1) * self.get(t)['reward'] for t in
                    range(update_time + 1, min(update_time + self.n, self.T) + 1)])

    def on_new_state(self, state, action, reward, next_state, done):
        if not self.train:
            return
        if self.t == 0:
            self.memory[self.t] = (state, action, 0)
        if self.t < self.T:
            self.get(self.t)['action'] = action
            self._store(self.t + 1, next_state, -1, reward)
            if done:
                self.T = self.t + 1

        update_time = self.t - self.n + 1
        if update_time >= 0:
            retruns = self._n_step_return(update_time)
            mem = self.get(update_time)
            mem_next_state = 0 if done else self.get(update_time + self.n)['state']
            self.model.train(mem['state'], mem['action'], retruns, mem_next_state, done)
        self.t += 1
        self.epsilon_decay.step()
        if done and update_time != self.T - 1:
            self.on_new_state(state, action, reward, next_state, done)
        elif done:
            self._reset()

    def on_episode_done(self, steps):
        pass

    def _store(self, t, state, action, reward):
        self.memory[t % (self.n + 1)] = (state, action, reward)

    def get(self, t):
        return self.memory[t % (self.n + 1)]


class EnvRunner(InterruptableProcess):
    def __init__(self, name, agent, episode_queue, delay):
        super().__init__()
        self.name = name
        self.agent = agent
        self.episode_queue = episode_queue
        self.delay = delay

    def run(self):
        print('[{}] Runner started'.format(self.pid))
        env = gym.make(self.name)
        while not self.interrupted.is_set():
            steps = generate_episode(env, self.agent, delay=self.delay)
            self.episode_queue.put(steps)
        print('[{}] Runner stopped'.format(self.pid))


class Predictor(InterruptableThread):
    def __init__(self, server, model, state_size, predict_queue, batch_size):
        super().__init__(daemon=True)
        self.server = server
        self.model = model
        self.state_size = state_size
        self.predict_queue = predict_queue
        self.batch_size = batch_size

    def _predict(self, ids, states):
        results = self.model.predict(states)
        for uid, *result in zip(ids, *results):
            self.server.get_agent_response_queue(uid).put(result)

    def run(self):
        print('[{}] Predictor started'.format(self.name))
        ids = np.zeros(self.batch_size, dtype=np.int)
        states = np.zeros((self.batch_size, self.state_size))
        current = 0
        while not self.interrupted:
            uid, state = self.predict_queue.get()
            ids[current] = uid
            states[current] = state
            current += 1
            if current >= self.batch_size or self.predict_queue.empty():
                self._predict(ids[:current], states[:current])
                current = 0
        print('[{}] Predictor stopped'.format(self.name))


class Trainer(InterruptableThread):
    def __init__(self, model: A3CModel, state_size, train_queue, batch_size):
        super().__init__(daemon=True)
        self.model = model
        self.state_size = state_size
        self.train_queue = train_queue
        self.batch_size = batch_size

    def run(self):
        print('[{}] Trainer started'.format(self.name))
        states = np.zeros((self.batch_size, self.state_size), np.float32)
        actions = np.zeros((self.batch_size, 1), np.int32)
        rewards = np.zeros((self.batch_size, 1), np.float32)
        next_states = np.zeros((self.batch_size, self.state_size), np.float32)
        dones = np.zeros((self.batch_size, 1), np.int32)
        current = 0
        while not self.interrupted:
            state, action, reward, next_state, done = self.train_queue.get()
            states[current] = state
            actions[current] = action
            rewards[current] = reward
            next_states[current] = next_state
            dones[current] = 0 if done else 1  # if done we need to omit last component of n-step return
            current += 1
            if current >= self.batch_size:
                self.model.train(states, actions, rewards, next_states, dones)
                current = 0
        print('[{}] Trainer stopped'.format(self.name))


class Server(InterruptableProcess):
    def __init__(self, env_name, state_size, action_size, n_agents, n_predictors, n_trainers, batch_size, n_episodes,
                 alpha, alpha_decay, gamma, c_v, c_e, clip_epsilon, n_epochs, n, epsilon_min, epsilon_max, epsilon_lam,
                 step_delay=0.001):
        super().__init__()
        self.env_name = env_name
        self.state_size = state_size
        self.action_size = action_size
        self.n_agents = n_agents
        self.n_predictors = n_predictors
        self.n_trainers = n_trainers
        self.batch_size = batch_size
        self.n_episodes = n_episodes
        self.alpha = alpha
        self.alpha_decay = alpha_decay
        self.gamma = gamma
        self.c_v = c_v
        self.c_e = c_e
        self.clip_epsilon = clip_epsilon
        self.n_epochs = n_epochs
        self.n = n
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.epsilon_lam = epsilon_lam
        self.step_delay = step_delay
        self.episode_queue = Queue(maxsize=100)
        self.predict_queue = Queue(maxsize=100)
        self.train_queue = Queue(maxsize=100)
        self.agents = []
        self.predictors = []
        self.trainers = []

    def _create_model_proxy(self, uid):
        return ModelProxy(uid, self.predict_queue, self.train_queue, Queue(maxsize=1))

    def _init_agents(self):
        for uid in range(self.n_agents):
            model_proxy = self._create_model_proxy(uid)
            epsilon_decay = EpsilonDecay(self.epsilon_max, self.epsilon_min, self.epsilon_lam)
            agent = Agent(self.state_size, self.action_size, model_proxy, self.n, self.gamma, epsilon_decay)
            runner = EnvRunner(self.env_name, agent, self.episode_queue, self.step_delay)
            self.agents.append(runner)
            runner.start()

    def _interrupt_agents(self):
        for agent in self.agents:
            agent.interrupt()
            # agent.agent.model.response_queue.close()

    def _init_predictors(self, model):
        for _ in range(self.n_predictors):
            predictor = Predictor(self, model, self.state_size, self.predict_queue, self.batch_size)
            self.predictors.append(predictor)
            predictor.start()

    def _interrupt_predictors(self):
        for predictor in self.predictors:
            predictor.interrupt()
        # self.predict_queue.close()

    def _init_trainers(self, model):
        for _ in range(self.n_trainers):
            trainer = Trainer(model, self.state_size, self.train_queue, self.batch_size)
            self.trainers.append(trainer)
            trainer.start()

    def _interrupt_trainers(self):
        for trainer in self.trainers:
            trainer.interrupt()
        # self.train_queue.close()

    def get_agent_response_queue(self, uid):
        model_proxy = self.agents[uid].agent.model
        return model_proxy.response_queue

    @staticmethod
    def _save_model(model: Model):
        model.save_weights('cartpole_a3c.h5')

    def run(self):
        print('[{}] Server started'.format(self.pid))
        start_time = clock()
        episodes_done = 0
        model = PolicyGradientModel(self.state_size, self.action_size, self.alpha, self.alpha_decay, self.gamma, self.n,
                                    self.c_v, self.c_e, self.clip_epsilon, self.n_epochs)
        self._init_trainers(model)
        self._init_predictors(model)
        self._init_agents()
        win_steps_in_row = 0
        while not self.interrupted.is_set():
            steps = self.episode_queue.get()
            episodes_done += 1
            if episodes_done % 100 == 0:
                print('Episodes done: {:4}, steps: {:3}, time: {}s'.format(episodes_done, steps, clock() - start_time))
            if episodes_done >= self.n_episodes:
                self.interrupt()
            # if steps >= 500:
            #     win_steps_in_row += 1
            # else:
            #     win_steps_in_row = 0
            # if win_steps_in_row >= 3 * self.n_agents:
            #     # self._save_model(model.model)
            #     self.interrupt()
        self._interrupt_agents()
        self._interrupt_predictors()
        self._interrupt_trainers()
        self._perform_plays(model)
        # self.episode_queue.close()
        print('[{}] Server stopped({}s)'.format(self.pid, clock() - start_time))

    def _perform_plays(self, model):
        K.set_learning_phase(0)
        agent = Agent(state_space_size, action_space_size, model, 8, 0.99, EpsilonDecay(0.0, 0.0, 0.0), train=False)
        env = gym.make(environment_id)
        while True:
            steps = generate_episode(env, agent, render=True)
            print('Steps:', steps)


def extract_space_sizes(name):
    env = gym.make(name)
    state_size, action_size = env.observation_space.shape[0], env.action_space.n
    env.close()
    return state_size, action_size


if __name__ == '__main__':
    environment_id = 'CartPole-v1'
    state_space_size, action_space_size = extract_space_sizes(environment_id)
    server = Server(environment_id, state_space_size, action_space_size,
                    n_agents=16, n_predictors=4, n_trainers=1, batch_size=4000, n_episodes=10000,
                    alpha=1e-2, alpha_decay=0.99, gamma=0.99, c_v=0.5, c_e=0.01, clip_epsilon=0.2,
                    n_epochs=5, n=8, epsilon_max=0.4, epsilon_min=0.01, epsilon_lam=0.001, step_delay=0)
    server.start()
    server.join()
