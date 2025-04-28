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
    show=True,
    vmin=None,
    vmax=None,
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
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * scale, n_rows * scale), squeeze=False
    )

    for row_i in range(n_rows):
        for col_i in range(n_cols):
            ax = axs[row_i, col_i]
            ax.imshow(
                images[row_i][col_i],
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
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

    if show:
        plt.show()
    else:
        return fig, axs


def display_ims(
    images: List[Image.Image],
    scale=3,
    titles=None,
    title=None,
    show=True,
    vmin=None,
    vmax=None,
):
    result = display_ims_grid(
        [images],
        scale,
        col_titles=titles,
        row_titles=None,
        title=title,
        show=show,
        vmin=vmin,
        vmax=vmax,
    )

    if not show:
        fig, axs = result
        return fig, axs[0]

    return result


def view_pointcloud_orthographic(
    ax: Axes, points: Tensor, horizontal_dim=0, vertical_dim=2, s=0.01, label=None
):
    dim_names = ["x", "y", "z"]

    ax.scatter(x=points[:, horizontal_dim], y=points[:, vertical_dim], s=s, label=label)
    ax.set_aspect("equal")
    ax.set_xlabel(dim_names[horizontal_dim])
    ax.set_ylabel(dim_names[vertical_dim])


def display_vid(clip: VideoClip, height=300):
    clip.write_videofile(
        "__temp__.mp4",
        verbose=False,
        logger=None,
    )

    return Video("__temp__.mp4", embed=True, height=height)


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
