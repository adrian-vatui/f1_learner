import gym

if __name__ == "__main__":
    env = gym.make('CarRacing-v0')

    for i_episode in range(20):
        observation = env.reset()
        for t in range(1000):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
