from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from moviepy.editor import (
    ColorClip,
    CompositeVideoClip,
    ImageSequenceClip,
    TextClip,
    clips_array,
)

from text3d2video.utilities.ipython_utils import display_vid
from text3d2video.utilities.video_util import extend_clip_repeat


@dataclass
class VideoLabel:
    content: str = ""
    font_size: int = 20
    color: str = "White"
    bg_color: str = "Black"
    font: str = "Hack-Regular"


def add_label_to_clip(clip, label: VideoLabel, position=("left", "top")):
    text_clip = (
        TextClip(
            label.content,
            fontsize=label.font_size,
            color=label.color,
            font=label.font,
            align="West",
            bg_color=label.bg_color,
        )
        .set_position(position)
        .set_duration(clip.duration)
        .set_fps(clip.fps)
    )

    return CompositeVideoClip([clip, text_clip])


def add_title_to_clip(clip: ImageSequenceClip, title: str) -> ImageSequenceClip:
    title_clip = (
        TextClip(
            title,
            fontsize=30,
            color="Black",
            bg_color="white",
            font="Droid-Sans",
            align="Center",
        )
        .set_position(("center", "top"))
        .set_duration(clip.duration)
        .set_fps(clip.fps)
    )
    title_clip = title_clip.margin(
        top=10, left=10, right=10, bottom=10, color=(255, 255, 255)
    )

    # add margin
    # pylint: disable=no-member
    new_clip = clip.margin(top=title_clip.h, left=0, right=0, color=(255, 255, 255))
    new_clip = CompositeVideoClip([new_clip, title_clip])
    return new_clip


def add_ylabel_to_clip(clip: ImageSequenceClip, title: str) -> ImageSequenceClip:
    title_clip = (
        TextClip(
            title,
            fontsize=30,
            color="Black",
            bg_color="white",
            font="Droid-Sans",
            align="Center",
        )
        .rotate(90.01)
        .set_position(("left", "center"))
        .set_duration(clip.duration)
        .set_fps(clip.fps)
    )
    title_clip = title_clip.margin(
        top=10, left=10, right=10, bottom=10, color=(255, 255, 255)
    )

    # add margin
    # pylint: disable=no-member
    new_clip = clip.margin(left=title_clip.w, right=0, color=(255, 255, 255))
    new_clip = CompositeVideoClip([new_clip, title_clip])
    return new_clip


def insert_cols(grid, col_indices, cols):
    inserted_indices = []

    for col_idx, col in zip(col_indices, cols):
        if col_idx >= grid.shape[1] - 1:
            raise ValueError(
                f"col_idx {col_idx} is out of bounds for grid with shape {grid.shape}"
            )

        for idx in inserted_indices:
            if idx <= col_idx:
                col_idx += 1

        grid = np.insert(grid, col_idx + 1, col, axis=1)
        inserted_indices.append(col_idx)

    return grid


def make_gap_col(col, size=15):
    gap_clips = [
        ColorClip(size=(size, v.size[1]), color=(255, 255, 255), duration=v.duration)
        for v in col
    ]
    gap_clips = np.array(gap_clips)
    return gap_clips


def video_grid(
    clips: np.ndarray,
    x_labels: List[str] = None,
    y_labels: List[str] = None,
    col_gap_indices=None,
    col_gap_sizes=None,
    pad_video_lengths: bool = True,
):
    """
    Arrange a grid of moviepy clips into a single clip as a grid, with optional x and y labels
    """
    clips = clips.copy()

    if pad_video_lengths:
        max_duration = np.vectorize(lambda v: v.duration)(clips).max()
        clips = np.vectorize(lambda v: extend_clip_repeat(v, max_duration))(clips)

    if x_labels is not None:
        top_row = clips[0]
        for i, clip in enumerate(top_row):
            top_row[i] = add_title_to_clip(clip, x_labels[i])

    if y_labels is not None:
        left_col = clips[:, 0]
        for i, clip in enumerate(left_col):
            left_col[i] = add_ylabel_to_clip(clip, y_labels[i])

    if col_gap_indices is not None:
        col = clips[:, 0]

        if col_gap_sizes is not None:
            gap_cols = [make_gap_col(col, size) for size in col_gap_sizes]
        else:
            gap_cols = [make_gap_col(col) for _ in col_gap_indices]

        clips = insert_cols(clips, col_gap_indices, gap_cols)

    array_clip = clips_array(clips)

    return array_clip


def display_vids(
    clips: List[ImageSequenceClip], titles: List[str] = None, title=None, width=300
):
    videos = [clips]
    clip = video_grid(videos, x_labels=titles, pad_video_lengths=True)

    if title:
        clip = add_title_to_clip(clip, title)

    return display_vid(clip, width=width)


def group_into_array(
    entries: List,
    dim_key_fns: List[Callable],
    return_keys: bool = False,
):
    """
    Group a list of entries into a multi-dimensional array, where each dimension is defined by a key function.
    Requires that the entries and key-functions define a grid
    """

    # find all unique keys for each dimension
    all_keys = [set() for _ in dim_key_fns]
    for entry in entries:
        # for each dimension, obtain the entry value
        for dim, key_fun in enumerate(dim_key_fns):
            key = key_fun(entry)
            all_keys[dim].add(key)

    # sort each set of keys to a sorted list, to assign an index to each key
    all_keys = [sorted(list(vals)) for vals in all_keys]

    # create empty array
    dim_sizes = [len(vals) for vals in all_keys]
    array = np.empty(shape=dim_sizes, dtype=object)

    # assign each entry to the correct index in the array
    for entry in entries:
        keys = [key_fun(entry) for key_fun in dim_key_fns]
        indices = [all_keys[dim].index(key) for dim, key in enumerate(keys)]

        if array[tuple(indices)] is not None:
            raise ValueError(f"Duplicate entry at {keys}")

        array[tuple(indices)] = entry

    if return_keys:
        return array, all_keys

    return array


def array_labels(array: np.ndarray, row_fun: Callable, col_fun: Callable):
    row_labels = [row_fun(row[0]) for row in array]
    col_labels = [col_fun(array[0][i]) for i in range(array.shape[1])]
    return row_labels, col_labels
