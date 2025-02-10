import shutil
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as TF
from IPython.display import HTML, Video
from matplotlib.axes import Axes
from moviepy.editor import VideoClip
from PIL.Image import Image
from torch import Tensor

from text3d2video.feature_visualization import reduce_feature_map


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


def display_ims(images: List[Image], scale=2, titles=None):
    if titles is not None:
        assert len(titles) == len(images)

    if len(images) == 1:
        _, ax = plt.subplots(1, 1, figsize=(scale, scale))
        ax.imshow(images[0])
        ax.axis("off")
        if titles is not None:
            ax.set_title(titles[0])
        plt.show()
        plt.tight_layout()
        plt.show()
        return

    _, axs = plt.subplots(1, len(images), figsize=(len(images) * scale, scale))

    for i, im in enumerate(images):
        axs[i].imshow(im)
        axs[i].axis("off")
        if titles is not None:
            axs[i].set_title(titles[i])

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


def display_vid(clip: VideoClip, resolution=300):
    clip.write_videofile(
        "__temp__.mp4",
        verbose=False,
        logger=None,
    )

    return Video("__temp__.mp4", embed=True, width=resolution)


def display_vids(clips: List[VideoClip], prefix="../", width=300):
    # initialize tempdir
    temp_dir = Path("__temp__")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir()

    video_paths = []

    for i, clip in enumerate(clips):
        vid_path = str(temp_dir / f"vid_{i}.mp4")

        clip.write_videofile(
            vid_path,
            verbose=False,
            logger=None,
        )
        video_paths.append(vid_path)

    video_paths = [prefix + vid_path for vid_path in video_paths]

    video_tags = "".join(
        f'<video width="{width}" controls><source src="{v}" type="video/mp4"></video>'
        for v in video_paths
    )

    return HTML(f'<div style="display: flex; gap: 10px;">{video_tags}</div>')


def reduce_feature_maps(
    features_grid: List[List[Tensor]], share_row=False, share_col=False
):
    images = []
    for row in features_grid:
        row_ims = []
        for feature_map in row:
            reduced = reduce_feature_map(feature_map.cpu())
            row_ims.append(reduced)
        images.append(row_ims)

    return images

    # display_ims_grid(images, scale=scale)
