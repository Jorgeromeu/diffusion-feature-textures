from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Video
from matplotlib.axes import Axes
from moviepy.editor import ImageSequenceClip
from PIL.Image import Image
from torch import Tensor


def display_ims_grid(
    images: List[List[Image]],
    scale=2.5,
    col_titles=None,
    row_titles=None,
    title=None,
):
    images = images.copy()

    n_rows = len(images)
    n_cols = len(images[0])

    if row_titles is not None:
        assert len(row_titles) == n_rows

    if col_titles is not None:
        assert len(col_titles) == n_cols

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * scale, n_rows * scale))
    if not isinstance(axs, np.ndarray):
        axs = np.array([axs])

    axs = axs.reshape(n_rows, n_cols)

    for row_i in range(n_rows):
        for col_i in range(n_cols):
            ax = axs[row_i, col_i]
            ax.imshow(images[row_i][col_i])
            ax.set_xticks([])
            ax.set_yticks([])

            if row_i == 0 and col_titles is not None:
                ax.set_title(col_titles[col_i])

            if col_i == 0 and row_titles is not None:
                ax.set_ylabel(row_titles[row_i])

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    plt.show()


def display_ims(images: List[Image], scale=2):
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


def view_pointcloud_orthographic(
    ax: Axes, points: Tensor, horizontal_dim=0, vertical_dim=2, s=0.01, label=None
):
    dim_names = ["x", "y", "z"]

    ax.scatter(x=points[:, horizontal_dim], y=points[:, vertical_dim], s=s, label=label)
    ax.set_aspect("equal")
    ax.set_xlabel(dim_names[horizontal_dim])
    ax.set_ylabel(dim_names[vertical_dim])
