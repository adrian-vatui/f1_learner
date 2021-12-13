import gym

from basic_actor import BasicActor
from ddpg_actor import DDPGActor

episodes_num = 10_000
RENDER = True
SKIP_FRAMES = 2

if __name__ == "__main__":
    env = gym.make('CarRacing-v0', verbose=0)
    # agent = BasicActor(buffer_size=10_000, batch_size=64)
    agent = DDPGActor(buffer_size=20_000)
    best_reward = -9999

    for i_episode in range(episodes_num):
        state = env.reset()
        done = False
        total_reward = 0
        negative_counter = 0
        skipped_frames = 0
        skipped_frames_reward = 0

        # skip the first 40 frames when the zooming happens
        for i in range(40):
            state, _, _, _ = env.step([0, 0, 0])
        
        

        while not done:
            if RENDER:
                env.render()

            skipped_frames += 1
            if skipped_frames == 1: 
                action, network_output = agent.get_action(state, training=True)  # get the action from the Neural network
                state_before_skip = state
            
            new_state, reward, done, info = env.step(action)  # execute the action

            #total_reward += reward

            # if reward < 0:
            #     reward *=2 # Increase negative reward for stepping on the grass
            skipped_frames_reward += reward

            if skipped_frames == SKIP_FRAMES: 
                agent.add_to_buffer(state_before_skip, network_output, skipped_frames_reward, new_state)  # add the data to the memory Buffer
                agent.train()  # and train the networks with the new data added in the memory buffer
                skipped_frames_reward = 0
                skipped_frames = 0

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

        print(f"[training] Finished episode {i_episode + 1} with reward {total_reward}")
