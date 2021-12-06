import gym

from networks import BasicNetwork

if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    agent = BasicNetwork()

    while True:
        state = env.reset()
        done = False

        while not done:
            env.render()
            action = agent.get_action(state)  # get the action from the Neural network
            new_state, reward, done, info = env.step(action)  # execute the action
            state = new_state
