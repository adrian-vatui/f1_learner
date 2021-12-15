import gym

from basic_actor import BasicActor
from ddpg_actor import DDPGActor

if __name__ == '__main__':
    env = gym.make('CarRacing-v0', verbose=0)
    # agent = BasicActor()
    agent = DDPGActor()
    agent.load(path='bestConfig2/')
    total_reward = 0
    episodes_num = 30

    for i_episode in range(episodes_num):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            env.render()
            action, network_output = agent.get_action(state, training=False)  # get the action from the Neural network
            new_state, reward, done, info = env.step(action)  # execute the action
            state = new_state
            episode_reward += reward

        total_reward += episode_reward

        print(f"[testing] Finished episode {i_episode + 1} with reward {episode_reward}")

    print("Average reward:", total_reward / episodes_num)
# python C:\ProgramData\Anaconda3\envs\f1_learner\Lib\site-packages\gym\envs\box2d\car_racing.py
