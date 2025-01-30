from dataclasses import dataclass
from typing import Callable, List

import numpy as np
from moviepy.editor import CompositeVideoClip, ImageSequenceClip, TextClip, clips_array

import text3d2video.wandb_util as wbu
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.ipython_utils import transpose_list_of_lists
from text3d2video.util import group_list_by
from wandb.apis.public import Run


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


def make_comparison_vid(
    runs: List[List[Run]],
    info_fun_top: Callable[[Run, VideoArtifact], VideoLabel] = None,
    info_fun_bottom: Callable[[Run, VideoArtifact], VideoLabel] = None,
    title: str = None,
    download=True,
):
    # ensure vids is rectangular
    for row in runs:
        if len(row) != len(runs[0]):
            raise ValueError("All rows must have the same number of runs")

    clips_grid = []
    for row in runs:
        row_clips = []

        for run in row:
            video_artifact = wbu.first_logged_artifact_of_type(run, "video")

            if video_artifact is None:
                print(f"Skipping {run.name} - no video artifact found")
                continue

            video_artifact = VideoArtifact.from_wandb_artifact(
                video_artifact, download=download
            )
            clip = video_artifact.get_moviepy_clip()

            if info_fun_bottom is not None:
                clip = add_label_to_clip(
                    clip,
                    info_fun_bottom(run, video_artifact),
                    position=("left", "bottom"),
                )

            if info_fun_top is not None:
                clip = add_label_to_clip(
                    clip, info_fun_top(run, video_artifact), position=("left", "top")
                )

            row_clips.append(clip)

        clips_grid.append(row_clips)

    array_clip = clips_array(clips_grid)

    if title is not None:
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
            .set_duration(array_clip.duration)
            .set_fps(array_clip.fps)
        )
        title_clip = title_clip.margin(
            top=10, left=10, right=10, bottom=10, color=(255, 255, 255)
        )

        # add margin
        # pylint: disable=no-member
        array_clip = array_clip.margin(
            top=title_clip.h, left=0, right=0, color=(255, 255, 255)
        )

        array_clip = CompositeVideoClip([array_clip, title_clip])

    return array_clip


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


def video_grid(
    clips: List[List[ImageSequenceClip]],
    title: str = None,
    x_labels: List[str] = None,
    y_labels: List[str] = None,
):
    """
    Arrange a grid of moviepy clips into a single clip as a grid, with optional x and y labels
    """

    # ensure vids is rectangular
    for row in clips:
        assert len(row) == len(clips[0]), "All rows must have the same number of clips"

    clips = clips.copy()

    # add x-labels
    if x_labels is not None:
        assert len(x_labels) == len(
            clips[0]
        ), "Number of x_labels must match number of clips"

        titled_top_clips = []
        for i, clip in enumerate(clips[0]):
            titled = add_title_to_clip(clip, x_labels[i])
            titled_top_clips.append(titled)
        clips[0] = titled_top_clips

    # add y-labels
    if y_labels is not None:
        clips = transpose_list_of_lists(clips)
        if y_labels is not None:
            titled_top_clips = []
            for i, clip in enumerate(clips[0]):
                titled = add_ylabel_to_clip(clip, y_labels[i])
                titled_top_clips.append(titled)
        clips[0] = titled_top_clips
        clips = transpose_list_of_lists(clips)

    array_clip = clips_array(clips)

    if title is not None:
        array_clip = add_title_to_clip(array_clip, title)

    return array_clip


def runs_grid(runs: List[List[Run]], x_label_fun=None, y_label_fun=None, labels=True):
    """
    Arrange a grid of runs into a single moviepy clip, with optional x and y labels
    """

    # get top row and left col to get labels
    x_labels = None
    if x_label_fun is not None:
        col_runs = runs[0]
        x_labels = [x_label_fun(r) for r in col_runs]

    y_labels = None
    if y_label_fun is not None:
        row_runs = [group[0] for group in runs]
        y_labels = [y_label_fun(r) for r in row_runs]

    vids = []
    for group in runs:
        group_vids = []
        for r in group:
            vid = wbu.first_logged_artifact_of_type(
                r, VideoArtifact.wandb_artifact_type
            )
            vid = VideoArtifact.from_wandb_artifact(vid)
            group_vids.append(vid.get_moviepy_clip())
        vids.append(group_vids)

    if not labels:
        x_labels = None
        y_labels = None

    return video_grid(vids, x_labels=x_labels, y_labels=y_labels)


def group_and_sort(
    entries: List,
    group_fun: Callable,
    sort_x_fun: Callable = None,
    sort_y_fun: Callable = None,
):
    # group by group_fun
    groups = group_list_by(entries, group_fun)

    # horizontal sort
    if sort_x_fun is not None:
        for group in groups:
            group.sort(key=sort_x_fun)

    # vertical sort
    if sort_y_fun is not None:
        groups.sort(key=lambda group: sort_y_fun(group[0]))

    return np.array(groups, dtype=object)
