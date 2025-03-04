from pathlib import Path
from typing import List

from IPython.display import Video
from moviepy.editor import VideoFileClip
from PIL import Image

import wandb_util.wandb_util as wbu
from text3d2video.utilities.video_util import clip_to_pil_frames, pil_frames_to_clip


class VideoArtifact(wbu.ArtifactWrapper):
    wandb_artifact_type = "video"

    def get_mp4_path(self) -> Path:
        return self.folder / "video.mp4"

    def write_frames(self, frames: List, fps=10):
        clip = pil_frames_to_clip(frames, fps=fps)
        clip.write_videofile(str(self.get_mp4_path()), codec="h264", bitrate="50000k")

    def get_moviepy_clip(self) -> VideoFileClip:
        return VideoFileClip(str(self.get_mp4_path()))

    def get_frames(self):
        clip = self.get_moviepy_clip()
        return clip_to_pil_frames(clip)

    def ipy_display(self):
        return Video(str(self.get_mp4_path()), embed=True)

    def get_frame_nums(self):
        log_run = self.logged_by()
        return log_run.config["frame_indices"]
