from pathlib import Path
from typing import List

import numpy as np
from IPython.display import Video
from moviepy.editor import ImageSequenceClip, VideoFileClip
from PIL import Image

import text3d2video.wandb_util as wu
from text3d2video.artifacts.animation_artifact import AnimationArtifact


def pil_frames_to_clip(frames: List, fps=10) -> ImageSequenceClip:
    # convert PIL images to numpy arrays
    frames_np = [np.asarray(im) for im in frames]

    # create video
    clip = ImageSequenceClip(frames_np, fps=fps)
    return clip


class VideoArtifact(wu.ArtifactWrapper):
    wandb_artifact_type = "video"

    def write_frames(self, frames: List, fps=10):
        clip = pil_frames_to_clip(frames, fps=fps)
        clip.write_videofile(str(self.get_mp4_path()))

    def get_mp4_path(self) -> Path:
        return self.folder / "video.mp4"

    def get_moviepy_clip(self) -> VideoFileClip:
        return VideoFileClip(str(self.get_mp4_path()))

    def get_frames(self):
        clip = self.get_moviepy_clip()
        frames = []
        for frame in clip.iter_frames():
            frames.append(Image.fromarray(frame))
        return frames

    def ipy_display(self):
        return Video(str(self.get_mp4_path()), embed=True)

    def get_frame_nums(self):
        log_run = self.logged_by()
        return log_run.config["frame_indices"]

    def get_animation_from_lineage(self):
        log_run = self.logged_by()
        animation = wu.first_used_artifact_of_type(
            log_run, AnimationArtifact.wandb_artifact_type
        )
        return AnimationArtifact.from_wandb_artifact(animation)
