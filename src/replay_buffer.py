from collections import namedtuple
from typing import List

class ReplayBuffer():
    def __init__(self, buffer_size : int = 100000):
        '''
        Creates a replay buffer of finite size to store previous state-action-reward-next action tuples

        args:
            buffer_size : int: the capacity of the replay buffer, 100,000 by default.
        '''
        self.capacity = buffer_size
        self.entry = namedtuple('Entry', ['state', 'action', 'reward', 'next_state']) # allows calling members by name instead of index 
        self.buffer : List[self.entry] = []

    def add_entry(self, state : List[float], action : float | list[float], reward : float, next_state : List[float]) -> None:
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
        self.buffer.append(self.entry(state, action, reward, next_state))
    
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

    def __getitem__(self, index: int) -> self.entry:
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