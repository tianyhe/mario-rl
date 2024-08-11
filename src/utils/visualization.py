"""
This module contains utility functions for visualizing data.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import display, HTML
import numpy as np


def display_frames_as_gif(frames, interval=50):
    """
    Displays a list of frames as a gif, with controls.

    Args:
    frames (list): List of frames to display, where each frame is a numpy array.
    interval (int): Time between frames in milliseconds. Default is 50.
    """
    fig, ax = plt.subplots()

    # Determine if frames are grayscale or RGB
    is_gray = len(frames[0].shape) == 2 or (
        len(frames[0].shape) == 3 and frames[0].shape[2] == 1
    )

    if is_gray:
        patch = ax.imshow(frames[0], cmap="gray")
    else:
        patch = ax.imshow(frames[0])

    plt.axis("off")

    def animate(i):
        if is_gray:
            patch.set_data(frames[i])
        else:
            patch.set_data(frames[i])
        return (patch,)

    anim = animation.FuncAnimation(
        fig, animate, frames=len(frames), interval=interval, blit=True
    )
    html = anim.to_html5_video()
    display(HTML(html))
    plt.close(fig)  # Close the figure to prevent it from displaying statically
