from pathlib import Path
from typing import List

import tempfile
import matplotlib.pyplot as plt
from PIL.Image import Image
import numpy as np
from moviepy.editor import ImageSequenceClip
from IPython.display import Video


def display_ims_grid(images: List[List[Image]], scale=1):

    n_rows = len(images)
    n_cols = len(images[0])

    _, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * scale, n_rows * scale))

    for row_i in range(n_rows):
        for col_i in range(n_cols):
            axs[row_i, col_i].imshow(images[row_i][col_i])
            axs[row_i, col_i].axis("off")

    plt.tight_layout()
    plt.show()


def display_ims(images: List[Image], scale=1):

    if len(images) == 1:
        _, ax = plt.subplots(1, 1, figsize=(scale, scale))
        ax.imshow(images[0])
        ax.axis("off")
        plt.show()
        plt.tight_layout()
        plt.show()
        return

    _, axs = plt.subplots(1, len(images), figsize=(len(images) * scale, scale))

    for i, im in enumerate(images):
        axs[i].imshow(im)
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()


def display_frames_as_video(frames: List[Image], path: Path, fps=10):

    # convert PIL images to numpy arrays
    frames_np = [np.asarray(im) for im in frames]

    # create video
    clip = ImageSequenceClip(frames_np, fps=fps)

    # write video to tempfile
    clip.write_videofile(str(path))
    clip.close()

    return Video(path.absolute())
