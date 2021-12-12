import numpy as np
import matplotlib.pyplot as plt
import time


def preprocess(state, greyscale=True):
    state = state.copy()  # making a copy so the image used in the function call doesn't get modified

    # Remove numbers and enlarge speed bar
    for i in range(88, 93 + 1):
        state[i, 0:12, :] = state[i, 12, :]

    # Make the car black
    car_color = 68.0
    car_area = state[67:77, 42:53]
    car_area[car_area == car_color] = 0

    # set the same color for the grass
    state = np.where(state == (102, 229, 102), (102, 204, 102), state)

    if not greyscale:
        return state

    # convert to grayscale
    state = np.dot(state[..., :3], [0.2989, 0.5870, 0.1140])
    state = np.expand_dims(state, axis=-1)

    # divide the value by 255
    state = state / 255

    # set the same color for the road
    state[(state > 0.411) & (state < 0.412)] = 0.4
    state[(state > 0.419) & (state < 0.420)] = 0.4

    # plt.imshow(state, cmap='gray')
    # plt.show()
    # time.sleep(1)
    return state


class Buffer:
    """
    Class for maintaining a history of visited states. Implemented like this instead of using (de)queues because
    by using numpy arrays it leads to less copying/casting/zipping
    """

    def __init__(self, capacity=50000):
        self.capacity = capacity
        self.states = None
        self.actions = None
        self.rewards = None
        self.new_states = None

        self.pos = 0  # index in array where the write operation is performed
        self.full = False  # true if the buffer is full

    def build_arrays(self, state_shape, action_shape):
        self.states = np.empty((self.capacity, *state_shape), dtype=np.float32)
        self.actions = np.empty((self.capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.new_states = np.empty((self.capacity, *state_shape), dtype=np.float32)

    def add(self, state, action, reward, new_state):
        if self.states is None:
            self.build_arrays(state.shape, action.shape)

        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.new_states[self.pos] = new_state

        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, k):
        # generate k random numbers in 0...pos (or capacity if full)
        random_indexes = np.random.choice(self.capacity if self.full else self.pos, k)
        return (self.states[random_indexes], self.actions[random_indexes],
                self.rewards[random_indexes], self.new_states[random_indexes])
