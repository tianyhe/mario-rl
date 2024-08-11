"""
Replay buffer for storing experiences and sampling for training.
"""

import torch
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
        self, state_size, action_size, buffer_size, batch_size, priority=False
    ):
        """Initialize a ReplayBuffer object."""
        self.states = torch.zeros((buffer_size,) + state_size).to(device)
        self.next_states = torch.zeros((buffer_size,) + state_size).to(device)
        self.actions = torch.zeros(buffer_size, 1, dtype=torch.long).to(device)
        self.rewards = torch.zeros(buffer_size, 1, dtype=torch.float).to(device)
        self.dones = torch.zeros(buffer_size, 1, dtype=torch.float).to(device)

        # Initialize self.e as a 1-dimensional array
        self.e = np.zeros(buffer_size, dtype=np.float64) if priority else None

        self.priority = priority
        self.ptr = 0
        self.n = 0
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        self.states[self.ptr] = torch.from_numpy(state).to(device)
        self.next_states[self.ptr] = torch.from_numpy(next_state).to(device)
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done

        if self.priority:
            self.e[self.ptr] = (
                1.0  # Assign initial priority, you can customize this value
            )

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.n = min(self.n + 1, self.buffer_size)

    def sample(self, get_all=False):
        """Randomly sample a batch of experiences from memory."""
        n = len(self)
        if get_all:
            return (
                self.states[:n],
                self.actions[:n],
                self.rewards[:n],
                self.next_states[:n],
                self.dones[:n],
            )

        if self.priority:
            # Normalize priorities before sampling
            p = self.e[:n] / self.e[:n].sum()
            idx = np.random.choice(n, self.batch_size, replace=False, p=p)
        else:
            idx = np.random.choice(n, self.batch_size, replace=False)

        states = self.states[idx]
        next_states = self.next_states[idx]
        actions = self.actions[idx]
        rewards = self.rewards[idx]
        dones = self.dones[idx]

        return (states, actions, rewards, next_states, dones), idx

    def update_error(self, e, idx=None):
        # Reduce the 2D tensor `e` to a 1D tensor by taking the mean or sum across the second dimension
        e = torch.abs(e.detach()).mean(
            dim=1
        )  # or use `.sum(dim=1)` depending on your requirement
        e = e / e.sum()

        if idx is not None:
            self.e[idx] = e.cpu().numpy()
        else:
            self.e[: len(self)] = e.cpu().numpy()

    def __len__(self):
        return self.n if self.n > 0 else self.ptr
