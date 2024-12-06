import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from gymnasium.spaces import Space

# Constants for weight initialization
FINAL_WEIGHT_INIT = 3e-3
FINAL_BIAS_INIT = 3e-4

class Actor(nn.Module):
    def __init__(self, hidden_size: List[int], num_inputs: int, action_space: Space)-> None:
        """
        Initializes the Actor class.

        Args:
            hidden_size (List[int,int]): The list of number of neurons in the hidden layers.
            num_inputs (int): The number of input features.
            action_space (Space): The size of the action space.
        """
        super(Actor, self).__init__()
        self.action_space = action_space
        output_size = action_space.shape[0]
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        self.ln2 = nn.LayerNorm(hidden_size[1])

        self.mu = nn.Linear(hidden_size[1], output_size)  # Output action space size

        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize weights for the neural network layers using fan-in uniform initialization."""

        # Layer 1 Initialization
        fan_in = self.linear1.weight.size(-1)
        w = 1. / np.sqrt(fan_in)
        nn.init.uniform_(self.linear1.weight, -w, w)
        nn.init.uniform_(self.linear1.bias, -w, w)

        # Layer 2 Initialization
        fan_in = self.linear2.weight.size(-1)
        w = 1. / np.sqrt(fan_in)
        nn.init.uniform_(self.linear2.weight, -w, w)
        nn.init.uniform_(self.linear2.bias, -w, w)

        # Output Layer Initialization
        nn.init.uniform_(self.mu.weight, -FINAL_WEIGHT_INIT, FINAL_WEIGHT_INIT)
        nn.init.uniform_(self.mu.bias, -FINAL_BIAS_INIT, FINAL_BIAS_INIT)


    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Actor network.

        Args:
            inputs (torch.Tensor): The input tensor containing state features.

        Returns:
            torch.Tensor: The output tensor representing actions in the range [-1, 1].
        """
        x = self.linear1(inputs)
        x = self.ln1(x)
        x = F.relu(x)

        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        return torch.tanh(self.mu(x))  # Bound actions in range [-1, 1]


class Critic(nn.Module):
    """
    Critic neural network for evaluating the value of state-action pairs in reinforcement learning.

    Args:
        hidden_size (int): The number of units in the hidden layers.
        num_inputs (int): The number of input features (state dimensions).
        action_space_size (int): The size of the action space.

    Methods:
        forward(inputs, actions):
            Performs a forward pass through the network.
            Args:
                inputs (torch.Tensor): The input state features.
                actions (torch.Tensor): The actions taken.
            Returns:
                torch.Tensor: The estimated value of the state-action pair.
    """

    def __init__(self, hidden_size: List[int], num_inputs: int, action_space: Space) -> None:
        super(Critic, self).__init__()
        self.action_space = action_space
        output_size = action_space.shape[0]

        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.ln1 = nn.LayerNorm(hidden_size[0])

        self.linear2 = nn.Linear(hidden_size[0] + output_size, hidden_size[1])  # Concatenate state and action
        self.ln2 = nn.LayerNorm(hidden_size[1])

        self.value = nn.Linear(hidden_size[1], 1)  # Output a single value
        self.initialize_weights()

    def initialize_weights(self) -> None:
        """Initialize weights for the neural network layers using fan-in uniform initialization."""

        # Layer 1 Initialization
        fan_in = self.linear1.weight.size(-1)
        w = 1. / np.sqrt(fan_in)
        nn.init.uniform_(self.linear1.weight, -w, w)
        nn.init.uniform_(self.linear1.bias, -w, w)

        # Layer 2 Initialization
        fan_in = self.linear2.weight.size(-1)
        w = 1. / np.sqrt(fan_in)
        nn.init.uniform_(self.linear2.weight, -w, w)
        nn.init.uniform_(self.linear2.bias, -w, w)

        # Output Layer Initialization
        nn.init.uniform_(self.value.weight, -FINAL_WEIGHT_INIT, FINAL_WEIGHT_INIT)
        nn.init.uniform_(self.value.bias, -FINAL_BIAS_INIT, FINAL_BIAS_INIT)


    def forward(self, inputs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the Critic network.

        Args:
            inputs (torch.Tensor): The input tensor containing state features.
            actions (torch.Tensor): The input tensor containing action features.

        Returns:
            torch.Tensor: The output tensor representing the value of the state-action pair.
        """
        x = self.linear1(inputs)
        x = self.ln1(x)
        x = F.relu(x)

        # Concatenate state features and actions
        x = torch.cat([x, actions], dim=1)

        x = self.linear2(x)
        x = self.ln2(x)
        x = F.relu(x)

        return self.value(x)  # Output value