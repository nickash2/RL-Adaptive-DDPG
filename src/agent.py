from copy import deepcopy
from utils.models import Actor, Critic
from gymnasium.spaces import Space
from typing import List
import torch
from utils.replay_buffer import ReplayBuffer
from utils.noise import OUNoise, NormalNoise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



class DDPG():
    def __init__(self, discount_factor: float, update_factor: float, action_space: Space, hidden_size: List[int,int], input_size: int, tau: float):
        self.discount_factor = discount_factor
        self.update_factor = update_factor
        self.action_space = action_space
        self.tau = tau

        self.actor = Actor(hidden_size, input_size, action_space)
        self.target_actor = Actor(hidden_size, input_size, action_space)

        self.critic = Critic(hidden_size, input_size, action_space)
        self.target_critic = Critic(hidden_size, input_size, action_space)

        self.replay_buffer = ReplayBuffer()

    def train(self):
        pass


    def select_action(self, state: torch.FloatTensor, noise = None):
        x = state.to(device)

        self.actor.eval()  # Sets the actor in evaluation mode
        mu = self.actor(x)  # Forward pass

        self.actor.train()  # Sets the actor back in training mode
        mu = mu.data

        if noise is not None:
            noise = torch.Tensor(noise.noise()).to(device)  # assuming noise object has func noise() that returns noise
            mu += noise

        # Clip the output according to the action space
        return mu.clamp(self.action_space.low[0], self.action_space.high[0])


    def soft_update(self, target, source, tau):
        """
        Soft update implementation (change this for proposed modification)
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


    def update(self):
        pass

    