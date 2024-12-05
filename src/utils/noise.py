import numpy as np

'''
Ornstein-Uhlenbeck action noise implementation for DDPG based upon:
https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
'''

class OUNoise():
    """
    Ornstein-Uhlenbeck process for generating noise, typically used in DDPG.

    Attributes:
        action_space (gym.Space): The action space of the environment.
        mu (float): The mean of the noise.
        theta (float): The rate of mean reversion.
        sigma (float): The volatility parameter.
        dt (float): The time step for the noise process.
        state (np.ndarray): The current state of the noise process.
    """

    def __init__(self, action_space, dt=0.01, mu=0, theta=0.15, sigma=0.2):
        """
        Initialize the OUNoise object.

        Args:
            action_space (gym.Space): The action space of the environment.
            dt (float): The time step for the noise process.
            mu (float): The mean of the noise.
            theta (float): The rate of mean reversion.
            sigma (float): The volatility parameter.
        """
        self.action_space = action_space
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()
    
    def reset(self):
        """
        Reset the state of the noise process to the mean value.
        """
        self.state = np.ones(self.action_space.shape) * self.mu

    def noise(self) -> np.ndarray:
        """
        Generate the next noise value using the Ornstein-Uhlenbeck process.

        Returns:
            np.ndarray: The next noise value.
        """
        x = self.state
        dx = self.theta * (self.mu - x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=x.shape)
        self.state = x + dx
        return self.state


class NormalNoise:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma

    def noise(self):
        return np.random.normal(self.mu, self.sigma)

    def __repr__(self):
        return 'NormalActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)