"""
Neural network models for the agent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetworkBase(nn.Module):
    """Base class for Q-Networks"""

    def __init__(self, channels, action_size, seed=42, input_shape=(4, 84, 84)):
        super(QNetworkBase, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(channels, 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2, ceil_mode=True)

        # Dynamically calculate flat_len based on input shape
        self.flat_len = self.calculate_flat_len(input_shape)

    def calculate_flat_len(self, input_shape):
        with torch.no_grad():
            sample_input = torch.zeros(1, *input_shape)
            output = self.conv_forward(sample_input)
            return output.shape[1]

    def conv_forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        return x.reshape(x.shape[0], -1)


class QNetworkCNN(QNetworkBase):
    """Traditional DQN network with CNN layers"""

    def __init__(self, channels, action_size, seed=42):
        super(QNetworkCNN, self).__init__(channels, action_size, seed)
        self.fc1 = nn.Linear(self.flat_len, 20)
        self.fc2 = nn.Linear(20, action_size)

    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = self.conv_forward(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class QNetworkDuellingCNN(QNetworkBase):
    """Duelling DQN network with CNN layers"""

    def __init__(self, channels, action_size, seed=42):
        super(QNetworkDuellingCNN, self).__init__(channels, action_size, seed)
        self.fcval = nn.Linear(self.flat_len, 20)
        self.fcval2 = nn.Linear(20, 1)
        self.fcadv = nn.Linear(self.flat_len, 20)
        self.fcadv2 = nn.Linear(20, action_size)

    def forward(self, x):
        """Build a network that maps state -> action values."""
        x = self.conv_forward(x)
        advantage = F.relu(self.fcadv(x))
        advantage = self.fcadv2(advantage)
        advantage = advantage - torch.mean(advantage, dim=-1, keepdim=True)
        value = F.relu(self.fcval(x))
        value = self.fcval2(value)
        return value + advantage


def create_q_network(network_type, channels, action_size, seed=42):
    """Factory function to create the desired Q-Network"""
    if network_type == "dqn":
        return QNetworkCNN(channels, action_size, seed)
    elif network_type == "duelling":
        return QNetworkDuellingCNN(channels, action_size, seed)
    else:
        raise ValueError(f"Unknown network type: {network_type}")
