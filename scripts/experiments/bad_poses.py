from dataclasses import dataclass

from hydra.compose import compose
from hydra.initialize import initialize
from omegaconf import OmegaConf

import scripts.run_generative_rendering
import text3d2video.wandb_util as wbu
from scripts.run_generative_rendering import ModelConfig, RunGenerativeRenderingConfig
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.artifacts.gr_data import GrSaveConfig
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.evaluation.video_comparison import video_grid
from text3d2video.experiment_util import WandbExperiment, object_to_instantiate_config
from text3d2video.generative_rendering.configs import (
    AnimationConfig,
    GenerativeRenderingConfig,
    RunConfig,
)
from text3d2video.noise_initialization import RandomNoiseInitializer
from text3d2video.rendering import render_depth_map
from text3d2video.video_util import pil_frames_to_clip

"""
Example experiment which runs generative rendering on a set of scenes and prompts
"""


@dataclass
class BadPosesExperimentConfig:
    model: ModelConfig
    run: RunConfig
    prompts: list[str]
    animations: list[AnimationConfig]
    save_tensors: GrSaveConfig
    generative_rendering: GenerativeRenderingConfig


class BadPosesExperiment(WandbExperiment):
    experiment_name = "bad_poses"

    def __init__(self):
        self.run_fn = scripts.run_generative_rendering.run

    def run_configs(self):
        configs = []

        with initialize(version_base=None, config_path="../../config"):
            config: BadPosesExperiment = compose(config_name=self.experiment_name)

        for anim in config.animations:
            for prompt in config.prompts:
                run = config.run
                cfg = RunGenerativeRenderingConfig(
                    run=run,
                    out_artifact="video",
                    prompt=prompt,
                    animation=anim,
                    generative_rendering=config.generative_rendering,
                    save_tensors=config.save_tensors,
                    noise_initialization=object_to_instantiate_config(
                        RandomNoiseInitializer()
                    ),
                    model=config.model,
                )

                cfg = OmegaConf.structured(cfg)

                configs.append(cfg)

        return configs

    def video_comparison(self):
        runs = self.get_logged_runs()

        depth_videos = []
        videos = []

        for run in runs:
            config: RunGenerativeRenderingConfig = OmegaConf.create(run.config)
            n_frames = config.animation.n_frames

            # get animation
            animation = wbu.first_used_artifact_of_type(run, "animation")
            animation = AnimationArtifact.from_wandb_artifact(animation)
            frame_indices = animation.frame_indices(n_frames)
            cams, meshes = animation.load_frames(frame_indices)

            # get depth maps
            depth_maps = render_depth_map(meshes, cams)

            # get video
            video_artifact = wbu.first_logged_artifact_of_type(run, "video")
            video_artifact = VideoArtifact.from_wandb_artifact(video_artifact)
            video_frames = video_artifact.get_moviepy_clip()

            # print(len(depth_maps))
            # print(len(video_frames))

            depth_video = pil_frames_to_clip(depth_maps, fps=10)
            # video_clip = pil_frames_to_clip(video_frames[0:-2], fps=10)

            depth_videos.append(depth_video)
            videos.append(video_frames)

        comparison_vid = video_grid([depth_videos, videos])
        return comparison_vid
