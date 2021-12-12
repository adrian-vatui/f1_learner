import gym

from basic_actor import BasicActor
from ddpg_actor import DDPGActor

episodes_num = 10_000
RENDER = True

if __name__ == "__main__":
    env = gym.make('CarRacing-v0')
    # agent = BasicActor(buffer_size=10_000, batch_size=64)
    agent = DDPGActor(buffer_size=10_000)
    best_reward = -9999

    for i_episode in range(episodes_num):
        state = env.reset()
        done = False
        total_reward = 0
        negative_counter = 0

        # skip the first 40 frames when the zooming happens
        for i in range(40):
            state, _, _, _ = env.step([0, 0, 0])

        while not done:
            if RENDER:
                env.render()
            action, network_output = agent.get_action(state)  # get the action from the Neural network
            new_state, reward, done, info = env.step(action)  # execute the action
            agent.add_to_buffer(state, network_output, reward, new_state)  # add the data to the memory Buffer
            agent.train()  # and train the networks with the new data added in the memory buffer

            # frame skips by performing the action again?

            total_reward += reward
            state = new_state

            if reward < 0:
                negative_counter += 1
            else:
                negative_counter = 0
            if negative_counter > 150:  # abandon the episode if I get negative rewards for too many consecutive frames
                break

        if total_reward > best_reward:  # save the best solution
            agent.save()
            best_reward = total_reward
