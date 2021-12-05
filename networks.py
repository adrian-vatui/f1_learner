import random
from collections import deque

import numpy as np
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import Sequential


# util https://keras.io/examples/rl/ddpg_pendulum/


class BasicNetwork:
    def __init__(
            self,
            action_space=np.array([(0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0.8)]),
            buffer_size=100,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.9999,
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

    # def create_actor(self, state_shape, name):
    #     input = layers.Input(shape = state_shape)
    #     #analyze the input state(the image) using a convolutional neural network
    #     x = input
    #     # create layers that with a 4x4 convolution window, that moves with the stride 4x4 
    #     # !!! We'll test other configurations later
    #     x = layers.Conv2D(64, kernel_size=(4,4), strides=(4,4),activation="relu", use_bias=False, padding="valid")(x)
    #     x = layers.Conv2D(32, kernel_size=(4,4), strides=(4,4),activation="relu", use_bias=False, padding="valid")(x)
    #     x = layers.Conv2D(32, kernel_size=(4,4), strides=(4,4),activation="relu", use_bias=False, padding="valid")(x)

    #     x = layers.Flatten()(x)
    #     x = layers.Dense(64,activation="relu")(x)
    #     # output is one of the possible actions
    #     y = layers.Dense(self.action_space.shape[0], activation='softmax')(x)

    #     model = Model(inputs = input, outputs=y, name=name)
    #     return model

    def build_model(self, state_shape):
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(7, 7), strides=(3, 3), activation='relu', input_shape=state_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), strides=(1, 1), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(216, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error")
        return model

        # input = layers.Input(shape=state_shape)
        # x = input
        # x = layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(4, 4), activation='relu')(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        # x = layers.Conv2D(filters=32, kernel_size=(5, 5), strides=(4, 4), activation='relu')(x)
        # x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        #
        # x = layers.Flatten()(x)
        # x = layers.Dense(64, activation='relu')(x)
        # y = layers.Dense(len(self.action_space), activation=None)(x)
        #
        # model = Model(inputs=input, outputs=y)
        # model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss="mean_squared_error")
        # return model

    # def create_critic(self, state_shape, name):
    #     input = layers.Input(shape = state_shape)
    #     #analyze the input state(the image) using a convolutional neural network
    #     x = input
    #     x = layers.Conv2D(64, kernel_size=(4,4), strides=(4,4),activation="relu", use_bias=False, padding="valid")(x)
    #     x = layers.Conv2D(32, kernel_size=(4,4), strides=(4,4),activation="relu", use_bias=False, padding="valid")(x)
    #     x = layers.Conv2D(32, kernel_size=(4,4), strides=(4,4),activation="relu", use_bias=False, padding="valid")(x)

    #     x = layers.Flatten()(x)
    #     actions_input = layers.Input(shape = (self.action_space.shape[0],))
    #     # combine the output after analyzing the input state(the image) with the possible actions, to get a reward..
    #     x = layers.concatenate([x, actions_input]) 
    #     x = layers.Dense(64,activation="relu")(x)
    #     x = layers.Dense(64,activation="relu")(x)
    #     y = layers.Dense(1)(x) # output is a single value (the reward)

    #     model = Model(inputs = [input, actions_input], outputs=y, name = name)
    #     return model

    # def initialize_networks_data(self, state_shape):
    #     #create both normal, and target networks
    #     self.actor = self.create_actor(state_shape, name='Actor')
    #     self.target_actor = self.create_actor(state_shape, name='TargetActor')

    #     self.critic = self.create_critic(state_shape, name='Critic')
    #     self.target_critic = self.create_critic(state_shape, name='TargetCritic')

    #     # ONLY at initialization set the same weights for the target networks
    #     self.target_actor.set_weights(self.actor.get_weights())
    #     self.target_critic.set_weights(self.target_critic.get_weights())

    def get_action(self, state):
        if self.model is None:
            self.model = self.build_model(state.shape)

        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(0, len(self.action_space))
        else:
            network_output = self.model.predict(state)
            action_index = np.argmax(network_output)

        return self.action_space[action_index]

    def add_to_buffer(self, state, action, reward, new_state):
        self.memory_buffer.append((state, action, reward, new_state))

    def train(self):
        if len(self.memory_buffer) < self.buffer_size:
            # not enough data to train yet, so we do nothing
            return

        batch = random.sample(self.memory_buffer, self.batch_size)
        states, targets = [], []
        for state, action, reward, new_state in batch:
            # feed-forward the current state
            target = self.model.predict(np.expand_dims(state, axis=0))[0]

            # get maximum Q-value for the next state
            next_q = np.amax(self.model.predict(np.expand_dims(new_state, axis=0))[0])

            # change only the target of the performed action
            target = np.where(target == action, reward + self.gamma * next_q, target)

            states.append(state)
            targets.append(target)

        # train the model
        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_best_solution(self, path='bestSolution/'):
        self.model.save(path + 'basic_model.h5')
