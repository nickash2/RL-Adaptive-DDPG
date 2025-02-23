from collections import namedtuple
from typing import List, Tuple
import random

class ReplayBuffer():
    def __init__(self, buffer_size : int = 100000, seed : int = 10):
        '''
        Creates a replay buffer of finite size to store previous state-action-reward-next action tuples

        args:
            buffer_size : int: the capacity of the replay buffer, 100,000 by default.
        '''
        self.capacity = buffer_size
        self.entry = namedtuple('Entry', ['state', 'action', 'reward', 'next_state', 'done']) # allows calling members by name instead of index 
        self.buffer : List[self.entry] = []
        random.seed(seed)

    def add_entry(self, state : List[float], action : float | list[float], reward : float, next_state : List[float], done : bool) -> None:
        '''
        Adds an entry during the learning phase of the model.

        Args:
            state : List[float]: a List of floats indicating the precise state of the agent and the environment.
            action : float | List[float]: The action neof the agent a single flaot for Inverted pendulum and a List of floats for Walker2-D
            reward: float: The reward obtained given the state and the action.
            next_state : List[float]: See state, changed by the action. 

        '''
        if len(self) >= self.capacity:
            self._remove_entry()
        self.buffer.append(self.entry(state, action, reward, next_state, done))
    
    def _remove_entry(self) -> None:
        '''
        Removes the first entry in the buffer
        
        Raises:
            IndexError: if the buffer is empty
        '''
        if not self.buffer:
            raise IndexError("There is no entry in this buffer to remove.")
        self.buffer.pop(0)

    def __len__(self) -> int:
        '''
        Returns the current length of the buffer
        '''
        return len(self.buffer)

    def __getitem__(self, index: int):
        '''
        Returns the member of the buffer at the given index, if present

        Args:
            index: the index of the element to be returned
        
        Raises:
            IndexError: if the requested index is too large for the buffer
        
        Returns:
            the requested element from the buffer
            
        '''
        if index >= len(self):
            raise IndexError("Index outside buffer.")
        return self.buffer[index]

    def clear(self) -> None:
        '''
        Empties the buffer
        '''
        self.buffer.clear()
    
    def sample(self, batch_size: int) -> Tuple[List[float], List[float], List[float], List[float], List[bool]]:
        '''
        Samples a random batch of entries from the replay buffer.

        Args:
            batch_size : int: The number of entries to sample.

        Raises:
            ValueError: If the requested batch size is greater than the number of entries in the buffer.

        Returns:
            Tuple[List[float], List[float], List[float], List[float], List[bool]]:
                - states: List of sampled states
                - actions: List of sampled actions
                - rewards: List of sampled rewards
                - next_states: List of sampled next states
                - dones: List of sampled done flags
        '''
        if batch_size > len(self):
            raise ValueError("Cannot sample more entries than are present in the buffer.")
        
        sampled_entries = random.sample(self.buffer, batch_size)

        states = [entry.state for entry in sampled_entries]
        actions = [entry.action for entry in sampled_entries]
        rewards = [entry.reward for entry in sampled_entries]
        next_states = [entry.next_state for entry in sampled_entries]
        dones = [entry.done for entry in sampled_entries]

        return states, actions, rewards, next_states, dones