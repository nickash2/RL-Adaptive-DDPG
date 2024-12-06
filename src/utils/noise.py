import numpy as np
from abc import ABC, abstractmethod

'''
Ornstein-Uhlenbeck action noise implementation for DDPG based upon:
https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
'''

class AbstractNoise(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def noise(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError


class OUNoise(AbstractNoise):
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


class NormalNoise(AbstractNoise):
    def __init__(self, mu=0, sigma=0.2, action_space=None, shape=None):
        """
        Normal Gaussian noise generator.

        Args:
            mu (float): Mean of the normal distribution.
            sigma (float): Standard deviation of the normal distribution.
            action_space (gym.Space, optional): Action space to determine noise shape.
            shape (tuple, optional): Explicit shape for the noise.
        """
        self.mu = mu
        self.sigma = sigma

        # Determine the shape of the noise
        if action_space is not None:
            self.shape = action_space.shape
        elif shape is not None:
            self.shape = shape
        else:
            raise ValueError("You must provide either an action_space or a shape for NormalNoise.")

        self.reset()

    def noise(self):
        """
        Generate noise.

        Returns:
            np.ndarray: Gaussian noise with the specified shape.
        """
        return np.random.normal(self.mu, self.sigma, size=self.shape)

    def reset(self):
        """
        Reset the state of the noise process.

        Returns:
            None: Keeps consistency with OUNoise reset.
        """
        # NormalNoise has no state to reset, but this adheres to the interface.
        self.state = np.zeros(self.shape)  # Consistency with OUNoise
