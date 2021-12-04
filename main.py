import re
import gym
import numpy as np

from noiseGenerator import *
from networks import *

episodes_num = 10000
render = True
action_space = [(-1, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1)] # define the action space

if __name__ == "__main__":
    env = gym.make('CarRacing-v0')
    agent = DPPGNetworks(action_space)
    best_reward = -9999

    for i_episode in range(episodes_num):
        current_state = env.reset()
        done = False
        total_reward = 0
        negative_counter = 0

        while not done:
            if render:
                env.render()
            state = agent.preprocess(state) # preprocess the image - GrayScale
            action = agent.get_action(current_state) # get the action from the Neural network
            new_state, reward, done, info = env.step(action) # do the action
            agent.add_to_buffer(state, action, reward, new_state) # add the data to the memory Buffer
            agent.train() # and train the networks with the new data added in the memory buffer

            total_reward += reward
            current_state = new_state

            if reward < 0:
                negative_counter += 1
            else:
                negative_counter  = 0
            if negative_counter > 100: # abandon the episode if I get negative rewards for 100 consecutive frames
                break
        
        if total_reward > best_reward: # save the best solution
            agent.save_best_solution()
            best_reward = total_reward