import sys

import gym

from ddpg_networks import DDPGNetwork

episodes_num = 10_000
RENDER = True

if __name__ == "__main__":
    env = gym.make('CarRacing-v0')
    agent = DDPGNetwork(env.action_space)
    best_reward = -9999

    for i_episode in range(episodes_num):
        state = env.reset()
        done = False
        total_reward = 0
        negative_counter = 0

        # skip the first few frames when the zooming animation happens
        for i in range(30):
            if RENDER:
                env.render()
            state, _, _, _ = env.step([0, 1, 0])

        while not done:
            if RENDER:
                env.render()
            action, network_action = agent.get_action(state)  # get the action from the Neural network
            new_state, reward, done, info = env.step(action)  # execute the action
            #agent.add_to_buffer(state, action, reward, new_state)  # add the data to the memory Buffer
            agent.train(state, network_action, reward, new_state)  # and train the networks with the new data added in the memory buffer

            total_reward += reward
            state = new_state

            if reward < 0:
                negative_counter += 1
            else:
                negative_counter = 0
            if negative_counter > 150:  # abandon the episode if I get negative rewards for 100 consecutive frames
                break

        if total_reward > best_reward:  # save the best solution
            agent.save_solution()
            best_reward = total_reward
