from typing import List

from attr import dataclass
from omegaconf import OmegaConf
from rerun import Image

import wandb_util.wandb_util as wbu
from scripts.wandb_runs.run_generative_rendering import RunGenerativeRenderingConfig
from text3d2video.artifacts.anim_artifact import AnimationArtifact, AnimSequence
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.clip_metrics import CLIPMetrics
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.utilities.video_comparison import VideoLabel, add_label_to_clip
from text3d2video.utilities.video_util import pil_frames_to_clip
from text3d2video.uv_consistency_metric import mean_uv_mse
from wandb.apis.public import Run


@dataclass
class VideoTraces:
    """
    Dataclass to hold all of the data relevant for analyzing a video generation run
    """

    frames: List[Image] = None
    prompt: str = None
    config: GenerativeRenderingConfig = None
    seq: AnimSequence = None
    uvs: List[Image] = None
    depths: List[Image] = None
    frame_consistency: float = None
    prompt_fidelity: float = None
    uv_mse: float = None

    @classmethod
    def from_run(
        cls, run: Run, with_video: bool = True, with_anim: bool = True
    ) -> "VideoTraces":
        data = cls()

        # Read video
        if with_video:
            video = wbu.logged_artifacts(run, "video")[0]
            video = VideoArtifact.from_wandb_artifact(video)
            data.frames = video.read_frames()

        # Read config
        # run_config: RunGenerativeRenderingConfig = OmegaConf.create(run.config)
        # data.config = run_config.generative_rendering

        # get prompt
        # data.prompt = run_config.prompt

        if with_anim:
            # read animation
            anim = wbu.used_artifacts(run, "animation")[0]
            anim = AnimationArtifact.from_wandb_artifact(anim)
            data.seq = anim.read_anim_seq()

        return data

    def compute_clip_metrics(self, model: CLIPMetrics):
        self.frame_consistency = model.frame_consistency(self.frames)
        self.prompt_fidelity = model.prompt_fidelity(self.frames, self.prompt)

    def compute_uv_mse(self):
        self.uv_mse = mean_uv_mse(
            self.frames,
            self.seq.cams,
            self.seq.meshes,
            self.seq.verts_uvs,
            self.seq.faces_uvs,
        )

    def video_with_metrics(self):
        clip = pil_frames_to_clip(self.frames)

        rows = []
        if self.frame_consistency is not None:
            rows.append(f"FC: {self.frame_consistency:.4f}")
        if self.prompt_fidelity is not None:
            rows.append(f"PF: {self.prompt_fidelity:.4f}")
        if self.uv_mse is not None:
            rows.append(f"UV MSE: {self.uv_mse:.4f}")

        content = "\n".join(rows).format(d=self)
        label = VideoLabel(content, font_size=30)
        return add_label_to_clip(clip, label, position=("left", "bottom"))
