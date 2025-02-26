from dataclasses import dataclass
from typing import List

from omegaconf import DictConfig, OmegaConf

import wandb_util.wandb_util as wbu
from scripts.run_generative_rendering import (
    ModelConfig,
    RunGenerativeRendering,
    RunGenerativeRenderingConfig,
)
from text3d2video.artifacts.anim_artifact import AnimationConfig
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.noise_initialization import UVNoiseInitializer
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.utilities.video_comparison import video_grid
from wandb_util.experiment_util import (
    object_to_instantiate_config,
)


@dataclass
class GrAblationConfig:
    model: ModelConfig
    generative_rendering: GenerativeRenderingConfig
    animation: AnimationConfig
    prompt: str


class GrAblation(wbu.Experiment):
    experiment_name = "gr_ablation"
    config: GrAblationConfig

    def __init__(self, config: DictConfig):
        self.config = config

    def specification(self) -> List[wbu.RunDescriptor]:
        noise_initialization = object_to_instantiate_config(UVNoiseInitializer())

        runs = []

        run_config = wbu.RunConfig(
            wandb=True,
            instant_exit=False,
            download_artifacts=False,
            name=None,
            group=None,
            tags=[],
        )

        gr_config = RunGenerativeRenderingConfig(
            run=run_config,
            prompt=self.config.prompt,
            animation=self.config.animation,
            generative_rendering=self.config.generative_rendering,
            model=self.config.model,
            noise_initialization=noise_initialization,
            seed=0,
        )

        gr_config = OmegaConf.create(gr_config)

        pre_only_config = gr_config.copy()
        pre_only_config.generative_rendering.do_post_attn_injection = False

        per_frame_config = pre_only_config.copy()
        per_frame_config.generative_rendering.do_pre_attn_injection = False

        gr_run = wbu.RunDescriptor(
            RunGenerativeRendering(), OmegaConf.structured(gr_config)
        )
        pre_only_run = wbu.RunDescriptor(
            RunGenerativeRendering(), OmegaConf.structured(pre_only_config)
        )
        per_frame_run = wbu.RunDescriptor(
            RunGenerativeRendering(), OmegaConf.structured(per_frame_config)
        )

        runs.append(gr_run)
        runs.append(pre_only_run)
        runs.append(per_frame_run)

        return runs

    def video_comparison(self):
        runs = self.get_logged_runs()

        def get_video(run):
            vid_artifact = wbu.first_logged_artifact_of_type(run, "video")
            vid_artifact = VideoArtifact.from_wandb_artifact(vid_artifact)
            return vid_artifact.get_moviepy_clip()

        videos = [get_video(run) for run in runs]
        return video_grid([videos])
