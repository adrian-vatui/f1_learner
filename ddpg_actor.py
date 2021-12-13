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
            buffer_size=10_000,
            gamma=0.99,
            batch_size=64,
            tau=0.005,
            noise_std=0.2
    ):
        self.noise_std = noise_std  # amount of noise added to network output in training
        self.tau = tau  # rate at which new actor/critic weights are applied to target actor/critic weights
        self.action_space_out = 2  # size of the output layer of the actor network
        self.buffer_size = buffer_size
        self.gamma = gamma  # how important future rewards are vs. the current reward
        self.batch_size = batch_size

        self.memory_buffer = utils.Buffer(capacity=self.buffer_size)
        self.actor = None
        self.critic = None
        self.target_actor = None
        self.target_critic = None
        self.actor_opt = Adam(0.0001)
        self.critic_opt = Adam(0.002)

    def create_actor(self, state_shape):
        input = layers.Input(shape=state_shape)
        x = input
        x = layers.Conv2D(16, kernel_size=5, strides=3, activation="relu", use_bias=False, padding="valid")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(32, kernel_size=3, strides=1, padding='valid', use_bias=False, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = layers.Flatten()(x)
        x = layers.Dense(64, activation="relu")(x)
        x = layers.GaussianNoise(self.noise_std)(x)
        # output is one of the possible actions
        y = layers.Dense(self.action_space_out, activation='tanh')(x)

        model = Model(inputs=input, outputs=y, name="Actor")
        model.summary()
        return model

    def create_critic(self, state_shape):
        input = layers.Input(shape=state_shape)
        # analyze the input state(the image) using a convolutional neural network
        x = input
        x = layers.Conv2D(16, kernel_size=5, strides=3, activation="relu", use_bias=False, padding="valid")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)
        x = layers.Conv2D(32, kernel_size=3, strides=1, padding='valid', use_bias=False, activation="relu")(x)
        x = layers.MaxPooling2D(pool_size=(2, 2))(x)

        x = layers.Flatten()(x)
        actions_input = layers.Input(shape=(self.action_space_out,))
        # combine the output after analyzing the input state(the image) with the possible actions, to get a reward..
        x = layers.Dense(64, activation="relu")(x)
        x = layers.concatenate([x, actions_input])
        x = layers.Dense(32, activation="relu")(x)
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

    def get_action(self, state, training=False):
        state = utils.preprocess(state)

        if self.actor is None:
            self.initialize_networks_data(state.shape)

        # Get the action from the actor network
        state = tf.expand_dims(tf.convert_to_tensor(state), 0)
        model_out = self.actor(state, training=training).numpy()

        model_out = model_out[0]
        network_action = model_out

        # print(model_out)
        # steer & gas or break
        model_out = np.array([np.clip(model_out[0], a_min=-1, a_max=1),
                              np.clip(model_out[1], a_min=0, a_max=1),
                              -np.clip(model_out[1], a_min=-1, a_max=0)])

        model_out[1] /= 1.25 # decrease speed even more
        return model_out / 3, network_action

    def add_to_buffer(self, state, action, reward, new_state):
        self.memory_buffer.add(utils.preprocess(state), action, reward, utils.preprocess(new_state))

    def train(self):
        state_batch, action_batch, reward_batch, new_state_batch = self.memory_buffer.sample(self.batch_size)

        state_batch = tf.convert_to_tensor(np.array(state_batch))
        action_batch = tf.convert_to_tensor(np.array(action_batch))
        reward_batch = tf.convert_to_tensor(np.array(reward_batch))
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        new_state_batch = tf.convert_to_tensor(np.array(new_state_batch))

        self.update_actor_critic(state_batch, action_batch, reward_batch, new_state_batch)

        # Update target networks
        self.update_target_network(self.target_actor.variables, self.actor.variables)
        self.update_target_network(self.target_critic.variables, self.critic.variables)

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
        for weight, target_weight in zip(weights, target_weights):
            target_weight.assign(weight * self.tau + target_weight * (1 - self.tau))

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
