import gym

from basic_actor import BasicActor
from ddpg_actor import DDPGActor

if __name__ == '__main__':
    env = gym.make('CarRacing-v0')
    # agent = BasicActor()
    agent = DDPGActor()
    agent.load()

    while True:
        state = env.reset()
        done = False

        while not done:
            env.render()
            action, network_output = agent.get_action(state, training=False)  # get the action from the Neural network
            new_state, reward, done, info = env.step(action)  # execute the action
            state = new_state
