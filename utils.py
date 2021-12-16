import numpy as np
import matplotlib.pyplot as plt
import time


def preprocess(state, greyscale=True):
    state = state.copy()  # making a copy so the image used in the function call doesn't get modified

    # Remove numbers and enlarge speed bar
    for i in range(88, 93 + 1):
        state[i, 0:12, :] = state[i, 12, :]

    # Make the car black
    # car_color = 68.0
    # car_area = state[67:77, 42:53]
    # car_area[car_area == car_color] = 0

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
    # state[(state > 0.411) & (state < 0.412)] = 0.4
    # state[(state > 0.419) & (state < 0.420)] = 0.4

    # plt.imshow(state, cmap='gray')
    # plt.show()
    # time.sleep(1)
    return state


def extract_features(state):
    # "metrics" part of the image starts at row 84 (inclusive)

    # speed: [93, 12:14] - [84, 12:14]
    speed = np.count_nonzero(np.mean(state[84:94, 12], axis=1))

    # all abs: [93, 17:26] - [84, 17:26]
    # abs1 and 2: [93, 17:22] - [84, 17:22]
    # abs1: [93, 17:19] - [84, 17:19]
    abs1 = np.count_nonzero(np.mean(state[84:94, 17], axis=1))

    # abs2: [93, 19:22] - [84, 19:22]
    abs2 = np.count_nonzero(np.mean(state[84:94, 19], axis=1))

    # abs3 and 4: [93, 22:26] - [84, 22:26]
    # abs3: [93, 22:24] - [84, 22:24]
    abs3 = np.count_nonzero(np.mean(state[84:94, 22], axis=1))

    # abs4: [93, 24:26] - [84, 24:26]
    abs4 = np.count_nonzero(np.mean(state[84:94, 24], axis=1))

    # steering: left: ? - 47, right: 48 - ?
    steering_left = np.count_nonzero(np.all(state[90][:48] == [0, 255, 0], axis=1))
    steering_right = np.count_nonzero(np.all(state[90][48:] == [0, 255, 0], axis=1))
    steering = -steering_left if steering_left > steering_right else steering_right

    # gyroscope: left: ? - 71, right: 72 - ?
    gyroscope_left = np.count_nonzero(np.all(state[90][:72] == [255, 0, 0], axis=1))
    gyroscope_right = np.count_nonzero(np.all(state[90][72:] == [255, 0, 0], axis=1))
    gyroscope = -gyroscope_left if gyroscope_left > gyroscope_right else gyroscope_right

    return np.array((speed, abs1, abs2, abs3, abs4, steering, gyroscope))


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
        self.extra_features = None
        self.new_extra_features = None

        self.pos = 0  # index in array where the write operation is performed
        self.full = False  # true if the buffer is full

    def build_arrays(self, state_shape, action_shape, extra_features_shape=None):
        self.states = np.empty((self.capacity, *state_shape), dtype=np.float32)
        self.actions = np.empty((self.capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((self.capacity, 1), dtype=np.float32)
        self.new_states = np.empty((self.capacity, *state_shape), dtype=np.float32)

        if extra_features_shape is not None:
            self.extra_features = np.empty((self.capacity, *extra_features_shape), dtype=np.float32)
            self.new_extra_features = np.empty((self.capacity, *extra_features_shape), dtype=np.float32)

    def add(self, state, action, reward, new_state, extra_features=None, new_extra_features=None):
        if self.states is None:
            if extra_features is None:
                self.build_arrays(state.shape, action.shape)
            else:
                self.build_arrays(state.shape, action.shape, extra_features.shape)

        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.new_states[self.pos] = new_state

        if extra_features is not None:
            self.extra_features[self.pos] = extra_features
            self.new_extra_features[self.pos] = new_extra_features

        self.pos = (self.pos + 1) % self.capacity
        if self.pos == 0:
            self.full = True

    def sample(self, k):
        # generate k random numbers in 0...pos (or capacity if full)
        random_indexes = np.random.choice(self.capacity if self.full else self.pos, k)

        if self.extra_features is None:
            return (self.states[random_indexes], self.actions[random_indexes],
                    self.rewards[random_indexes], self.new_states[random_indexes])

        return (self.states[random_indexes], self.actions[random_indexes],
                self.rewards[random_indexes], self.new_states[random_indexes],
                self.extra_features[random_indexes], self.new_extra_features[random_indexes])
