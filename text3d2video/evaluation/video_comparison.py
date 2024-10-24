from dataclasses import dataclass
from typing import Callable, List

from moviepy.editor import CompositeVideoClip, TextClip, clips_array
from omegaconf import OmegaConf

import text3d2video.wandb_util as wbu
from text3d2video.artifacts.animation_artifact import AnimationArtifact
from text3d2video.artifacts.video_artifact import VideoArtifact, pil_frames_to_clip
from text3d2video.rendering import render_depth_map
from wandb.apis.public import Run


@dataclass
class VideoLabel:
    content: str = ""
    font_size: int = 20
    color: str = "White"
    bg_color: str = "Black"
    font: str = "Hack-Regular"


def scene_key_fun(run: Run):
    cfg = OmegaConf.create(run.config)
    prompt = cfg.prompt
    animation = cfg.animation.artifact_tag
    n_frames = cfg.animation.n_frames
    return frozenset({prompt, animation, n_frames})


def make_comparison_vid(
    runs: List[List[Run]],
    info_fun: Callable[[Run], VideoLabel],
    title: str = None,
    download=True,
    show_guidance_video=False,
):
    # ensure vids is rectangular
    for row in runs:
        if len(row) != len(runs[0]):
            raise ValueError("All rows must have the same number of runs")

    clips_grid = []
    for row in runs:
        row_clips = []

        if show_guidance_video:
            row_run = row[0]
            animation = wbu.first_used_artifact_of_type(row_run, "animation")
            animation = AnimationArtifact.from_wandb_artifact(animation, download=download)

            n_frames = OmegaConf.create(row_run.config).animation.n_frames
            frame_nums = animation.frame_nums(n_frames)
            frames = animation.load_frames(frame_nums, device="cpu")
            cameras = animation.cameras(frame_nums)
            depth_maps = render_depth_map(frames, cameras)

            depth_video = pil_frames_to_clip(depth_maps, fps=10)
            row_clips.append(depth_video)

        for run in row:
            video_artifact = wbu.first_logged_artifact_of_type(run, "video")

            if video_artifact is None:
                print(f"Skipping {run.name} - no video artifact found")
                continue

            video_artifact = VideoArtifact.from_wandb_artifact(video_artifact, download=download)
            clip = video_artifact.get_moviepy_clip()

            label = info_fun(run)

            text_clip = (
                TextClip(
                    label.content,
                    fontsize=label.font_size,
                    color=label.color,
                    font=label.font,
                    align="West",
                    bg_color=label.bg_color,
                )
                .set_position(("left", "top"))
                .set_duration(clip.duration)
                .set_fps(clip.fps)
            )

            composited = CompositeVideoClip([clip, text_clip])

            row_clips.append(composited)

        clips_grid.append(row_clips)

    array_clip = clips_array(clips_grid)

    if title is not None:
        title_clip = (
            TextClip(
                title,
                fontsize=30,
                color="Black",
                bg_color="white",
                font="CMU-Serif",
                align="Center",
            )
            .set_position(("center", "top"))
            .set_duration(array_clip.duration)
            .set_fps(array_clip.fps)
        )
        title_clip = title_clip.margin(top=10, left=10, right=10, bottom=10, color=(255, 255, 255))

        # add margin
        # pylint: disable no-member
        array_clip = array_clip.margin(top=title_clip.h, left=0, right=0, color=(255, 255, 255))

        array_clip = CompositeVideoClip([array_clip, title_clip])

    return array_clip
