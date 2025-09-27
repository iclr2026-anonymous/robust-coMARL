"""
Experience replay buffer for Robust MARL algorithms.
"""

from collections import deque
import numpy as np

class ReplayBuffer:
    """
    Experience replay buffer using deque for efficient FIFO operations.
    """
    def __init__(self, capacity, rng=None):
        self.buffer = deque(maxlen=capacity)
        self.rng = rng if rng is not None else np.random.default_rng()

    def add(self, transition):
        """Add a transition to the buffer."""
        self.buffer.append(transition)

    def sample(self, batch_size):
        """Sample a batch of transitions from the buffer."""
        indices = self.rng.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.buffer)

    def is_ready(self, batch_size):
        """Check if the buffer has enough samples for training."""
        return len(self.buffer) >= batch_size 