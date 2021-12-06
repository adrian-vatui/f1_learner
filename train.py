import sys

import gym

from networks import BasicNetwork

episodes_num = 10_000
RENDER = True

if __name__ == "__main__":
    env = gym.make('CarRacing-v0')
    agent = BasicNetwork(buffer_size=10_000)
    best_reward = -9999

    for i_episode in range(episodes_num):
        state = env.reset()
        done = False
        total_reward = 0
        negative_counter = 0

        # skip the first few frames when the zooming animation happens
        for i in range(40):
            if RENDER:
                env.render()
            state, _, _, _ = env.step([0, 0.2, 0])

        while not done:
            if RENDER:
                env.render()
            action = agent.get_action(state)  # get the action from the Neural network
            new_state, reward, done, info = env.step(action)  # execute the action
            agent.add_to_buffer(state, action, reward, new_state)  # add the data to the memory Buffer
            agent.train()  # and train the networks with the new data added in the memory buffer

            total_reward += reward
            state = new_state

            # if reward < 0:
            #     negative_counter += 1
            # else:
            #     negative_counter = 0
            # if negative_counter > 100:  # abandon the episode if I get negative rewards for 100 consecutive frames
            #     break

        if total_reward > best_reward:  # save the best solution
            agent.save()
            best_reward = total_reward
