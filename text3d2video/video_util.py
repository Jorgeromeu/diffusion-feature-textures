from typing import List

import numpy as np
from moviepy.editor import ImageSequenceClip


def pil_frames_to_clip(frames: List, fps=10) -> ImageSequenceClip:
    # convert PIL images to numpy arrays
    frames_rgb = [im.convert("RGB") for im in frames]
    frames_np = [np.asarray(im) for im in frames_rgb]
    frames_np = frames_np + [frames_np[-1]]

    # create video
    clip = ImageSequenceClip(frames_np, fps=fps)
    return clip
