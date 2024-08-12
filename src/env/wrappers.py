"""
Environment wrappers for preprocessing the observations and actions.
"""

import gymnasium as gym
import numpy as np
import torch
from torchvision import transforms as T
from gymnasium.spaces import Box, Discrete


class CustomActionWrapper(gym.ActionWrapper):
    """
    Custom action wrapper for mapping the action space to a custom action space.
    """

    def __init__(self, env, action_mapping):
        super(CustomActionWrapper, self).__init__(env)
        self.action_mapping = action_mapping
        self.action_space = Discrete(len(action_mapping))  # Limit the action space

    def action(self, action):
        if action < len(self.action_mapping):
            return self.action_mapping[action]
        else:
            raise IndexError("Action index is out of range")


class SkipFrame(gym.Wrapper):
    """
    SkipFrame wrapper for skipping frames in the environment.
    """

    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    """
    GrayScaleObservation wrapper for converting the observation to grayscale.
    """

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


class ResizeObservation(gym.ObservationWrapper):
    """
    ResizeObservation wrapper for resizing the observation to a specific shape.
    """

    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape) if isinstance(shape, int) else tuple(shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        if isinstance(observation, np.ndarray):
            transforms = T.Compose(
                [
                    T.ToTensor(),
                    T.Resize(self.shape, antialias=True),
                    T.Normalize(0, 255),
                ]
            )
            observation = transforms(observation).squeeze(0)
            return observation.numpy()
        elif isinstance(observation, torch.Tensor):
            transforms = T.Compose([T.Resize(self.shape), T.Normalize(0, 255)])
            return transforms(observation).squeeze(0)
        else:
            raise TypeError(f"Unexpected observation type: {type(observation)}")
