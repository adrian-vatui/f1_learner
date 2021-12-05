import numpy as np


def preprocess(state, greyscale=False):
    # crop the image (Remove the bar below and 6 pixels from each margin)
    state = state[:-12, 6:-6]

    # make the color of the car black
    car_position = state[66:79, 36:47]
    car_position[car_position == 204] = 0

    # set the same color for the grass
    state = np.where(state == (102, 229, 102), (102, 204, 102), state)

    if not greyscale:
        return state

    # convert to grayscale
    np.dot(state[..., :3], [0.2989, 0.5870, 0.1140])
    # divide the value by 255
    state = state / 255

    # set the same color for the road
    state[(state > 0.411) & (state < 0.412)] = 0.4
    state[(state > 0.419) & (state < 0.420)] = 0.4

    # plt.imshow(state, cmap='gray')
    # plt.show()
    # time.sleep(5)
    return state
