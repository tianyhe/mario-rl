"""
This module contains the main classes and functions for the environment.
"""

from .wrappers import (
    CustomActionWrapper,
    SkipFrame,
    GrayScaleObservation,
    ResizeObservation,
)
from gymnasium.wrappers import FrameStack

# Action sets
RIGHT_ONLY = [0, 3, 6, 11, 14]
SIMPLE_MOVEMENT = [0, 3, 6, 11, 14, 2, 4]
COMPLEX_MOVEMENT = [0, 3, 6, 11, 14, 2, 4, 7, 12, 15, 5, 2]

# Version information
__version__ = "0.1.0"


def create_env(
    env_id,
    action_set=RIGHT_ONLY,
    skip_frames=4,
    gray_scale=True,
    resize_shape=84,
    stack_frames=4,
    full_action_space=False,
):
    import gymnasium as gym

    env = gym.make(env_id, render_mode="rgb_array")

    if not full_action_space:
        env = CustomActionWrapper(env, action_set)

    env = SkipFrame(env, skip=skip_frames)

    if gray_scale:
        env = GrayScaleObservation(env)

    env = ResizeObservation(env, shape=resize_shape)

    if stack_frames != 1:
        env = FrameStack(env, num_stack=4)
    else:
        return env

    return env


__all__ = [
    "CustomActionWrapper",
    "SkipFrame",
    "GrayScaleObservation",
    "ResizeObservation",
    "RIGHT_ONLY",
    "SIMPLE_MOVEMENT",
    "COMPLEX_MOVEMENT",
    "create_env",
]

# Run setup code
import logging

logging.getLogger(__name__).addHandler(logging.NullHandler())
