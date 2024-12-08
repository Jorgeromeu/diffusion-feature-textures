from typing import List

import numpy as np
from moviepy.editor import ImageSequenceClip


def duration_to_fps(duration: int, n_frames: int) -> int:
    return n_frames / duration


def pil_frames_to_clip(frames: List, fps=10, duration=None) -> ImageSequenceClip:
    # convert PIL images to numpy arrays
    frames_rgb = [im.convert("RGB") for im in frames]
    frames_np = [np.asarray(im) for im in frames_rgb]
    frames_np = frames_np + [frames_np[-1]]

    if duration is not None:
        fps = duration_to_fps(duration, len(frames_np))

    # create video
    clip = ImageSequenceClip(frames_np, fps=fps)
    return clip
