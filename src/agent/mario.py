"""
Agent module for Mario.
"""

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from agent.net import create_q_network
from agent.replay_buffer import ReplayBuffer
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def process_state(state):
    """
    Extracts the LazyFrames object from the state tuple and converts it to a NumPy array.
    """
    observation = np.array(state[0])  # Extract and convert LazyFrames to NumPy array
    return observation


class Mario:
    """Mario agent that interacts with and learns from the environment."""

    def __init__(
        self,
        state_dim,
        action_dim,
        save_dir,
        network_type="duelling",
        ddqn=True,
        priority=False,
        use_cuda=True,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.network_type = network_type
        self.save_dir = save_dir
        self.ddqn = ddqn
        self.device = torch.device(
            "cuda" if use_cuda and torch.cuda.is_available() else "cpu"
        )

        # Create the online network
        self.net = create_q_network(network_type, state_dim[0], action_dim).to(
            self.device
        )

        if self.ddqn:
            # Create the target network if using Double DQN
            self.target_net = create_q_network(
                network_type, state_dim[0], action_dim
            ).to(self.device)
            self.target_net.load_state_dict(
                self.net.state_dict()
            )  # Initialize with the same weights
            self.target_net.eval()  # Target network does not need gradients
        else:
            self.target_net = None  # No target network if not using DDQN

        self.exploration_rate = 1
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.save_every = 5e5  # no. of experiences between saving Mario Net

        self.memory = ReplayBuffer(state_dim, action_dim, int(1e5), 32, priority)
        self.batch_size = 32

        self.gamma = 0.9
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = nn.SmoothL1Loss()

        self.burnin = 1e4  # min. experiences before training
        self.learn_every = 3  # no. of experiences between updates to Q_online
        self.sync_every = 1e4  # no. of experiences between Q_target & Q_online sync

    def act(self, state, eps=0.0):
        """Returns actions for given state as per current policy."""
        if np.random.rand() < eps:
            action_idx = np.random.randint(self.action_dim)
        else:
            if isinstance(state, tuple):
                observation = np.array(state[0])
            else:
                observation = np.array(state)

            if observation.ndim == 2:
                observation = observation[np.newaxis, ...]

            state_tensor = torch.tensor(observation, device=self.device).unsqueeze(0)
            action_values = self.net(
                state_tensor
            )  # Use online network for action selection
            action_idx = torch.argmax(action_values, axis=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action_idx

    def cache(self, state, next_state, action, reward, done):
        """Add the experience to memory"""
        observation = np.array(state[0])
        next_observation = np.array(next_state[0])
        self.memory.add(observation, action, reward, next_observation, done)

    def recall(self):
        """Sample experiences from memory"""
        return self.memory.sample()

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        if self.ddqn and self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        (states, actions, rewards, next_states, dones), idx = self.recall()
        td_est = self.td_estimate(states, actions)
        td_tgt = self.td_target(rewards, next_states, dones)
        loss = self.update_Q_online(td_est, td_tgt)

        # print(f"td_estimate shape: {td_est.shape}")
        # print(f"td_target shape: {td_tgt.shape}")
        # print(f"e (td_estimate - td_target) shape: {(td_est - td_tgt).shape}")
        self.memory.update_error(td_est - td_tgt, idx)

        return (td_est.mean().item(), loss)

    def td_estimate(self, state, action):
        """Estimate the action value (Q) using the online network"""
        current_Q = self.net(state)[np.arange(0, self.batch_size), action.squeeze()]
        return current_Q

    @torch.no_grad()
    def td_target(self, reward, next_state, done):
        """Estimate the target Q-value using the target network"""
        if self.ddqn:
            # Double DQN logic: select action using the online network, evaluate with the target network
            next_state_Q = self.net(
                next_state
            )  # Use online network for action selection
            best_action = torch.argmax(
                next_state_Q, axis=1
            ).squeeze()  # Ensure best_action is 1D
            # Ensure next_Q is 1D by using the correct indexing
            next_Q = self.target_net(next_state)[
                np.arange(0, self.batch_size), best_action
            ].squeeze()
        else:
            # Standard DQN logic: both action selection and evaluation use the same network
            next_Q = self.net(next_state).max(dim=1)[
                0
            ]  # Use online network for the target Q-value

        # Ensure the resulting tensor is 1D
        return (reward + (1 - done.float()) * self.gamma * next_Q).float().squeeze()

    def update_Q_online(self, td_estimate, td_target):
        """Update the online action value (Q) function with a batch of experiences"""
        loss = self.loss_fn(td_estimate, td_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def sync_Q_target(self):
        """Synchronize the target network with the online network"""
        if self.target_net is not None:
            self.target_net.load_state_dict(self.net.state_dict())

    def save(self, filename=None):
        """Save the model"""
        if filename is None:
            # Use the original filename format if no filename is provided
            save_path = (
                self.save_dir
                / f"mario_net_{int(self.curr_step // self.save_every)}.chkpt"
            )
        else:
            # Use the provided filename
            save_path = Path(filename)

        # Ensure the directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            dict(
                model=self.net.state_dict(),
                exploration_rate=self.exploration_rate,
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                network_type=self.network_type,
                ddqn=self.ddqn,
                priority="priority" if self.memory.priority else "uniform",
            ),
            save_path,
        )
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")

    def load(self, filename):
        """
        Load a saved model state.

        Args:
            filename (str): Path to the saved model file.
        """
        checkpoint = torch.load(filename, map_location=self.device)

        # Load model parameters
        self.net.load_state_dict(checkpoint["model"])
        if self.ddqn and self.target_net is not None:
            self.target_net.load_state_dict(checkpoint["model"])

        # Load other attributes
        self.exploration_rate = checkpoint["exploration_rate"]

        # Verify that the loaded model matches the current instance
        assert self.state_dim == tuple(
            checkpoint["state_dim"]
        ), "State dimensions do not match"
        assert (
            self.action_dim == checkpoint["action_dim"]
        ), "Action dimensions do not match"
        assert (
            self.network_type == checkpoint["network_type"]
        ), "Network type does not match"
        assert self.ddqn == checkpoint["ddqn"], "DDQN setting does not match"

        # Handle priority setting
        loaded_priority = checkpoint["priority"] == "priority"
        if loaded_priority != self.memory.priority:
            print(
                f"Warning: Loaded model used {'priority' if loaded_priority else 'uniform'} replay, "
                f"but current instance uses {'priority' if self.memory.priority else 'uniform'} replay."
            )

        print(f"MarioNet loaded from {filename}")
