from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
from IPython.display import Video
from matplotlib.axes import Axes
from moviepy.editor import VideoClip
from PIL import Image
from torch import Tensor


def display_ims_grid(
    images: List[List[Image.Image]],
    scale=2.5,
    col_titles=None,
    row_titles=None,
    title=None,
):
    images = images.copy()

    # shape
    n_rows = len(images)
    n_cols = len(images[0])

    if row_titles is not None:
        assert len(row_titles) == n_rows

    if col_titles is not None:
        assert len(col_titles) == n_cols

    # make figure
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
            ax.set_frame_on(False)

            if row_i == 0 and col_titles is not None:
                ax.set_title(col_titles[col_i])

            if col_i == 0 and row_titles is not None:
                ax.set_ylabel(row_titles[row_i])

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    # plt.show()


def display_ims(
    images: List[Image.Image], scale=3, titles=None, title=None, vmin=None, vmax=None
):
    if titles is not None:
        assert len(titles) == len(images)

    if len(images) == 1:
        _, ax = plt.subplots(1, 1, figsize=(scale, scale))
        ax.imshow(images[0], vmin=vmin, vmax=vmax)
        ax.axis("off")
        if titles is not None:
            ax.set_title(titles[0])
        plt.show()
        plt.tight_layout()
        plt.show()
        return

    _, axs = plt.subplots(1, len(images), figsize=(len(images) * scale, scale))

    for i, im in enumerate(images):
        axs[i].imshow(im, vmin=vmin, vmax=vmax)
        axs[i].axis("off")
        if titles is not None:
            axs[i].set_title(titles[i])

    if title is not None:
        plt.suptitle(title)

    plt.tight_layout()
    plt.show()


def view_pointcloud_orthographic(
    ax: Axes, points: Tensor, horizontal_dim=0, vertical_dim=2, s=0.01, label=None
):
    dim_names = ["x", "y", "z"]

    ax.scatter(x=points[:, horizontal_dim], y=points[:, vertical_dim], s=s, label=label)
    ax.set_aspect("equal")
    ax.set_xlabel(dim_names[horizontal_dim])
    ax.set_ylabel(dim_names[vertical_dim])


def display_vid(clip: VideoClip, width=300):
    clip.write_videofile(
        "__temp__.mp4",
        verbose=False,
        logger=None,
    )

    return Video("__temp__.mp4", embed=True, width=width)


def to_pil_image(feature_map: torch.Tensor, clip=False):
    if clip:
        feature_map_in = feature_map.clamp(0, 1)
    else:
        feature_map_in = feature_map

    return Image.fromarray(
        (feature_map_in.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )


def max_divisor(n):
    if n <= 1:
        return None  # No proper divisor for 1 or less
    for i in range(n // 2, 0, -1):
        if n % i == 0:
            return i


def reshape_lst_to_rect(lst: List, divisor=None):
    if divisor is None:
        divisor = max_divisor(len(lst))

    chunked = [lst[i : i + divisor] for i in range(0, len(lst), divisor)]
    return chunked
