import numpy as np
from numpy.random.mtrand import random
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from collections import deque
from tensorflow.python.keras.backend import softmax

from tensorflow.python.keras.layers.pooling import MaxPool2D, MaxPooling2D

#util https://keras.io/examples/rl/ddpg_pendulum/

class DPPGNetworks:
    def __init__(self, action_space):
        self.action_space = action_space
        self.buffer_size  = 10000
        self.memoryBuffer = deque(maxlen = self.buffer_size)
        # self.actor          = None
        # self.critic         = None
        # self.target_actor   = None
        # self.target_critic  = None

        self.basic_model = None
        self.epsilon = 1.0


    
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

    def create_basic_model(self, state_shape):
        input = layers.Input(shape = state_shape)
        x = input
        x = layers.Conv2D(16, kernel_size=(5,5), strides=(4,4), activation='relu', use_bias=False, padding="valid")(x)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)
        x = layers.Conv2D(32, kernel_size=(5,5), strides=(4,4), activation='relu', use_bias=False, padding="valid")(x)
        x = layers.MaxPooling2D(pool_size=(2,2))(x)
        
        x = layers.Flatten()(x)
        x = layers.Dense(64, activation='relu')(x)
        y = layers.Dense(self.action_space.shape[0], activation='softmax')(x)
        
        model = Model(inputs = input, outputs = y)
        return model



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
        if self.basic_model == None:
            self.basic_model = self.create_basic_model(state.shape)
        
        if np.random.rand() < self.epsilon:
            action_index = np.random.randint(0,len(self.action_space))
        else:
            network_output = self.basic_model.predict(state)
            action_index = np.argmax(network_output)
        

    def add_to_buffer(self, state, action, reward, new_state):
        self.memoryBuffer.append((state, action, reward, new_state))

    def train(self):
        pass
    
    def preprocess(self, state):
        #crop the image (Remove the bar below and 6 pixels from each margin)
        state = state[:-12,6:-6]

        # make the color of the car black
        car_position = state[66:79, 36:47]
        car_position[car_position == 204] = 0

        #set the same color for the grass
        state = np.where(state == (102, 229, 102),(102, 204, 102), state)

        # convert to grayscale
        np.dot(state[...,:3], [0.2989, 0.5870, 0.1140]) 
        # divide the value by 255
        state = state / 255

        #set the same color for the road
        state[(state > 0.411) & (state < 0.412)] = 0.4
        state[(state > 0.419) & (state < 0.420)] = 0.4

        # plt.imshow(state, cmap='gray')
        # plt.show()
        # time.sleep(5)
        return state


    def save_best_solution(self, path = 'bestSolution/'):
        self.basic_model.save(path + 'basic_model.h5')




