from dataclasses import dataclass
from typing import List

import numpy as np
from IPython.display import Video
from moviepy.editor import (
    ColorClip,
    CompositeVideoClip,
    ImageSequenceClip,
    TextClip,
    VideoClip,
    clips_array,
    vfx,
)

from text3d2video.util import object_array
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


class VideoPadding:
    REPEAT = "repeat"
    SLOW_DOWN = "slow_down"
    ZERO = "zero"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False


def video_grid(
    clips: np.ndarray,
    x_labels: List[str] = None,
    y_labels: List[str] = None,
    col_gap_indices=None,
    col_gap_sizes=None,
    padding_mode: str = VideoPadding.REPEAT,
):
    """
    Arrange a grid of moviepy clips into a single clip as a grid, with optional x and y labels
    """
    clips = clips.copy()

    if padding_mode == VideoPadding.REPEAT:
        max_duration = np.vectorize(lambda v: v.duration)(clips).max()
        clips = np.vectorize(lambda v: extend_clip_repeat(v, max_duration))(clips)

    if padding_mode == VideoPadding.SLOW_DOWN:
        max_duration = np.vectorize(lambda v: v.duration)(clips).max()
        clips = np.vectorize(lambda v: v.fx(vfx.speedx, v.duration / max_duration))(
            clips
        )

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


def display_vid(clip: VideoClip, height=300, title=None):
    if title:
        clip = add_title_to_clip(clip, title)

    clip.write_videofile(
        "__temp__.mp4",
        verbose=False,
        logger=None,
    )

    return Video("__temp__.mp4", embed=True, height=height)


def display_vids(
    clips: List[ImageSequenceClip],
    titles: List[str] = None,
    title=None,
    height=300,
    match_height=True,
    padding_mode: str = VideoPadding.REPEAT,
    vertical=False,
):
    if match_height:
        max_height = max([v.size[1] for v in clips])
        clips = [v.resize(height=max_height) for v in clips]

    videos = object_array([clips])

    video_grid_kwargs = {
        "padding_mode": padding_mode,
    }

    if vertical:
        videos = np.transpose(videos)
        video_grid_kwargs["y_labels"] = titles
    else:
        video_grid_kwargs["x_labels"] = titles

    clip = video_grid(videos, **video_grid_kwargs)

    if title:
        clip = add_title_to_clip(clip, title)

    return display_vid(clip, height=height)
