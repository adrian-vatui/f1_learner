import numpy as np

# using Ornstein-Uhlenbeck process for generating noise to implement better exploration for the Actor
class OUNoiseGenerator:
    def __init__(self, mean, std_deviation, theta = 0.15, dt = 0.0001, x_initial = None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x = x_initial
        self.reset()
    
    def generator(self):
        self.x = self.x + self.theta *(self.mean - self.x) * self.dt \
                 + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        return self.x    

    def reset(self):
        if self.x is None:
            self.x = np.zeros_like(self.mean)