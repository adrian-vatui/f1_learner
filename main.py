import gym
import numpy as np

from noiseGenerator import *



if __name__ == "__main__":
    env = gym.make('CarRacing-v0')

    for i_episode in range(2):
        observation = env.reset()
        for t in range(1000):
            env.render()
            #action = env.action_space.sample()
            observation, reward, done, info = env.step([0.0, 1.0, 0.0])
