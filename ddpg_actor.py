import random
from collections import deque

import keras.models
import tensorflow as tf
import numpy as np
from keras import layers
from keras.optimizers import Adam

# util https://keras.io/examples/rl/ddpg_pendulum/
from tensorflow.python.keras import Model

import utils


class DDPGActor:
    def __init__(
            self,
            action_space=np.array([(0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0.8)]),
            buffer_size=10_000,
            gamma=0.95,
            batch_size=64,
            tau=0.005,
            epsilon=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.9999,
    ):
        self.tau = tau
        self.action_space = action_space
        self.action_space_out = 2
        self.buffer_size = buffer_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.memory_buffer = deque(maxlen=self.buffer_size)
        self.actor = None
        self.critic = None
        self.target_actor = None
        self.target_critic = None
        self.actor_opt = Adam(0.001)
        self.critic_opt = Adam(0.002)

    def create_actor(self, state_shape):
        input = layers.Input(shape=state_shape)
        x = input
        x = layers.Conv2D(16, kernel_size=(5, 5), strides=(4, 4), activation="relu", use_bias=False, padding="valid")(x)
        x = layers.Conv2D(32, kernel_size=(4, 4), strides=(4, 4), activation="relu", use_bias=False, padding="valid")(x)
        x = layers.Conv2D(32, kernel_size=(4, 4), strides=(4, 4), activation="relu", use_bias=False, padding="valid")(x)

        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        # output is one of the possible actions
        y = layers.Dense(self.action_space_out, activation='tanh')(x)

        model = Model(inputs=input, outputs=y, name="Actor")
        model.summary()
        return model

    def create_critic(self, state_shape):
        input = layers.Input(shape=state_shape)
        # analyze the input state(the image) using a convolutional neural network
        x = input
        x = layers.Conv2D(16, kernel_size=(5, 5), strides=(4, 4), activation="relu", use_bias=False, padding="valid")(x)
        x = layers.Conv2D(32, kernel_size=(4, 4), strides=(4, 4), activation="relu", use_bias=False, padding="valid")(x)
        x = layers.Conv2D(32, kernel_size=(4, 4), strides=(4, 4), activation="relu", use_bias=False, padding="valid")(x)

        x = layers.Flatten()(x)
        actions_input = layers.Input(shape=(self.action_space_out,))
        # combine the output after analyzing the input state(the image) with the possible actions, to get a reward..
        x = layers.concatenate([x, actions_input])
        x = layers.Dense(64, activation="relu")(x)
        x = layers.Dense(64, activation="relu")(x)
        y = layers.Dense(1)(x)

        model = Model(inputs=[input, actions_input], outputs=y)
        model.summary()
        return model

    def initialize_networks_data(self, state_shape):
        # create both normal, and target networks
        self.actor = self.create_actor(state_shape)
        self.target_actor = self.create_actor(state_shape)

        self.critic = self.create_critic(state_shape)
        self.target_critic = self.create_critic(state_shape)

        # ONLY at initialization set the same weights for the target networks
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.target_critic.get_weights())

    def get_action(self, state):
        state = utils.preprocess(state)

        if self.actor is None:
            self.initialize_networks_data(state.shape)

        # Get the action from the actor network
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        model_out = self.actor(state).numpy()

        model_out = model_out[0]
        network_action = model_out

        if np.random.rand() < self.epsilon:
            model_out = np.array(random.choice([(0, 0, 0), (-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 0.8)]))
            network_action = [model_out[0], model_out[1] - model_out[2]]
        else:
            # print(model_out)
            # steer & gas or break
            model_out = np.array([model_out[0], max(model_out[1], 0), max(-model_out[1], 0)])

        return model_out / 3, network_action

    def add_to_buffer(self, state, action, reward, new_state):
        self.memory_buffer.append((utils.preprocess(state), action, reward, utils.preprocess(new_state)))

    def train(self):
        if len(self.memory_buffer) >= self.batch_size:
            state_batch, action_batch, reward_batch, new_state_batch = zip(
                *random.sample(self.memory_buffer, self.batch_size))

            state_batch = tf.convert_to_tensor(np.array(state_batch))
            action_batch = tf.convert_to_tensor(np.array(action_batch))
            reward_batch = tf.convert_to_tensor(np.array(reward_batch))
            reward_batch = tf.cast(reward_batch, dtype=tf.float32)
            new_state_batch = tf.convert_to_tensor(np.array(new_state_batch))

            self.update_actor_critic(state_batch, action_batch, reward_batch, new_state_batch)

            # Update target networks
            self.update_target_network(self.target_actor.variables, self.actor.variables)
            self.update_target_network(self.target_critic.variables, self.critic.variables)

            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon * self.epsilon_decay

    @tf.function
    def update_actor_critic(self, state, action, reward, new_state):
        # Update critic
        with tf.GradientTape() as tape:
            new_action = self.target_actor(new_state, training=True)
            y = reward + self.gamma * self.target_critic([new_state, new_action], training=True)

            critic_value = self.critic([state, action], training=True)
            critic_loss = tf.math.reduce_mean(tf.square(y - critic_value))

        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_opt.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        # Update actor
        with tf.GradientTape() as tape:
            critic_out = self.critic([state, self.actor(state, training=True)], training=True)
            actor_loss = -tf.math.reduce_mean(critic_out)

        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_opt.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

    @tf.function
    def update_target_network(self, target_weights, weights):
        for a, b in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

    def save(self, path='bestSolution/'):
        self.actor.save(path + 'actor.h5')
        self.critic.save(path + 'critic.h5')
        self.target_actor.save(path + 'target_actor.h5')
        self.target_critic.save(path + 'target_critic.h5')

    def load(self, path='bestSolution/'):
        self.actor = keras.models.load_model(path + 'actor.h5')
        self.critic = keras.models.load_model(path + 'critic.h5')
        self.target_actor = keras.models.load_model(path + 'target_actor.h5')
        self.target_critic = keras.models.load_model(path + 'target_critic.h5')
