import random
import sys
from collections import deque

import keras.models
import numpy as np
from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from tensorflow.python.keras.layers import MaxPooling2D

import utils


# util https://keras.io/examples/rl/ddpg_pendulum/
# notite: functia de cost = -output critic

class BasicActor:
    def __init__(
            self,
            action_space=np.array([(0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0.8)]),
            buffer_size=10_000,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.999,
            learning_rate=0.001,
            batch_size=64
    ):
        self.action_space = action_space
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        self.memory_buffer = deque(maxlen=self.buffer_size)
        self.model = None

    def build_model(self, state_shape):
        model = Sequential()
        model.add(Conv2D(filters=16, kernel_size=5, strides=3, padding="valid", activation='relu',
                         input_shape=state_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=32, kernel_size=3, strides=1, padding="valid", activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(32, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error")
        model.summary()
        return model

    def get_action(self, state, training=False):
        state = utils.preprocess(state)

        if self.model is None:
            self.model = self.build_model(state.shape)

        if training and np.random.rand() < self.epsilon:
            action_index = np.random.randint(0, len(self.action_space))
        else:
            network_output = self.model.predict(np.expand_dims(state, axis=0))[0]
            action_index = np.argmax(network_output)

        return self.action_space[action_index], self.action_space[action_index]

    def add_to_buffer(self, state, action, reward, new_state):
        self.memory_buffer.append((utils.preprocess(state), action, reward, utils.preprocess(new_state)))

    def train(self):
        if len(self.memory_buffer) < self.buffer_size:
            # not enough data to train yet, so we do nothing
            return

        batch = random.sample(self.memory_buffer, self.batch_size)
        states, actions, rewards, new_states = zip(*batch)

        states = np.array(states)
        new_states = np.array(new_states)

        # feed-forward the current states
        targets = self.model(states).numpy()

        # get maximum Q-values for the next states
        next_qs = [np.amax(next_target) for next_target in self.model(new_states).numpy()]

        # iterate through targets and change only the targets of the performed action
        for target, action, reward, next_q in zip(targets, actions, rewards, next_qs):
            action_index = np.where(self.action_space == action)[0][0]
            target[action_index] = reward + self.gamma * next_q
        targets = np.array(targets)

        # train the model
        self.model.fit(states, targets, epochs=1, verbose=0, batch_size=self.batch_size)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path='bestSolution/'):
        self.model.save(path + 'basic_model.h5')

    def load(self, path='bestSolution/'):
        self.model = keras.models.load_model(path + 'basic_model.h5')
