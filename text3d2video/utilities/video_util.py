from typing import List

import numpy as np
from moviepy.editor import ImageSequenceClip, concatenate_videoclips
from PIL import Image


def duration_to_fps(duration: int, n_frames: int) -> int:
    return n_frames / duration


def pil_frames_to_clip(frames: List, fps=10, duration=None) -> ImageSequenceClip:
    # convert PIL images to numpy arrays
    frames_rgb = [im.convert("RGB") for im in frames]
    frames_np = [np.asarray(im) for im in frames_rgb]
    frames_np = frames_np

    if duration is not None:
        fps = duration_to_fps(duration, len(frames_np))

    # create video
    clip = ImageSequenceClip(frames_np, fps=fps)
    return clip


def clip_to_pil_frames(clip: ImageSequenceClip, expected_frames: int = None) -> List:
    frames = []

    if expected_frames is None:
        n_frames_float = clip.duration * clip.fps
        expected_frames = round(n_frames_float)

    for frame in clip.iter_frames():
        frames.append(Image.fromarray(frame))
    return frames[:expected_frames]


def extend_clip_to_match_duration(clip, target_duration):
    """
    Extends a MoviePy clip to match the duration of another video by repeating it.

    :param clip: The MoviePy clip to extend.
    :param target_duration: The target duration (in seconds) to match.
    :return: A new MoviePy clip extended to the target duration.
    """
    if clip.duration is None:
        raise ValueError("The input clip must have a defined duration.")

    if target_duration <= 0:
        raise ValueError("Target duration must be greater than 0.")

    # Calculate how many full repetitions are needed
    repeat_count = int(target_duration // clip.duration)
    remaining_duration = target_duration % clip.duration

    # Create repeated clips
    clips = [clip] * repeat_count
    if remaining_duration > 0:
        clips.append(clip.subclip(0, remaining_duration))

    # Concatenate the clips
    extended_clip = concatenate_videoclips(clips)

    return extended_clip
