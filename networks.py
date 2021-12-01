import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam

class Networks:
    def __init__(self, action_space):
        self.action_space = action_space
        self.actor          = None
        self.critic         = None
        self.target_actor   = None
        self.target_critic  = None

    def create_actor(self, state_shape, name="Actor"):
        input = layers.Input(shape = state_shape)
        x = input
        x = layers.Conv2D(32, kernel_size=(3,3), strides=(3,3),activation="relu", use_bias=False, padding="valid")(x)
        x = layers.Conv2D(32, kernel_size=(3,3), strides=(3,3),activation="relu", use_bias=False, padding="valid")(x)
        x = layers.Conv2D(32, kernel_size=(3,3), strides=(3,3),activation="relu", use_bias=False, padding="valid")(x)

        x = layers.Flatten()(x)
        x = layers.Dense(64,activation="relu")(x)
        y = layers.Dense(self.action_space.shape[0], activation='softmax')(x)

        model = Model(inputs = input, outputs=y, name=name)
        return model
    def create_critic(self, state_shape, name="Critic"):
        pass

    def create_target_actor(self, state_shape):
        pass
    def create_target_critic(self, state_shape):
        pass
    def initialize_networks_data(self, state_shape):
        pass



