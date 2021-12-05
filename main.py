import gym

from networks import *

episodes_num = 10000
RENDER = True

if __name__ == "__main__":
    env = gym.make('CarRacing-v0')
    agent = BasicNetwork()
    best_reward = -9999

    for i_episode in range(episodes_num):
        state = env.reset()
        done = False
        total_reward = 0
        negative_counter = 0

        while not done:
            if RENDER:
                env.render()
            # state = utils.preprocess(state, greyscale=False)  # preprocess the image
            action = agent.get_action(state)  # get the action from the Neural network
            new_state, reward, done, info = env.step(action)  # do the action
            agent.add_to_buffer(state, action, reward, new_state)  # add the data to the memory Buffer
            agent.train()  # and train the networks with the new data added in the memory buffer

            total_reward += reward
            current_state = new_state

            if reward < 0:
                negative_counter += 1
            else:
                negative_counter = 0
            if negative_counter > 100:  # abandon the episode if I get negative rewards for 100 consecutive frames
                break

        if total_reward > best_reward:  # save the best solution
            agent.save_best_solution()
            best_reward = total_reward
