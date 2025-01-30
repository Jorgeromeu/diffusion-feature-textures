from dataclasses import dataclass

import numpy as np
from hydra.compose import compose
from hydra.initialize import initialize
from omegaconf import OmegaConf

import scripts.run_generative_rendering
import text3d2video.utilities.wandb_util as wbu
from scripts.run_generative_rendering import ModelConfig, RunGenerativeRenderingConfig
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.artifacts.gr_data import GrSaveConfig
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.generative_rendering.configs import (
    AnimationConfig,
    GenerativeRenderingConfig,
    RunConfig,
)
from text3d2video.noise_initialization import RandomNoiseInitializer, UVNoiseInitializer
from text3d2video.rendering import render_depth_map
from text3d2video.utilities.experiment_util import (
    WandbExperiment,
    object_to_instantiate_config,
)
from text3d2video.utilities.video_comparison import (
    group_into_array,
    video_grid,
)
from text3d2video.utilities.video_util import pil_frames_to_clip

"""
Example experiment which runs generative rendering on a set of scenes and prompts
"""


@dataclass
class BadPosesExperimentConfig:
    model: ModelConfig
    run: RunConfig
    prompt: str
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
            config: BadPosesExperimentConfig = compose(config_name=self.experiment_name)

        for anim in config.animations:
            for per_frame in [True, False]:
                gr_conf: GenerativeRenderingConfig = config.generative_rendering.copy()

                if per_frame:
                    gr_conf.do_post_attn_injection = False
                    gr_conf.do_pre_attn_injection = False
                    noise_initialization = RandomNoiseInitializer()
                else:
                    noise_initialization = UVNoiseInitializer()

                run = config.run
                cfg = RunGenerativeRenderingConfig(
                    run=run,
                    out_artifact="video",
                    prompt=config.prompt,
                    animation=anim,
                    generative_rendering=gr_conf,
                    save_tensors=config.save_tensors,
                    noise_initialization=object_to_instantiate_config(
                        noise_initialization
                    ),
                    model=config.model,
                )

                cfg = OmegaConf.structured(cfg)

                configs.append(cfg)

        return configs

    def video_comparison(self, show_depth_videos=True):
        runs = self.get_logged_runs()

        def cfg(run) -> RunGenerativeRenderingConfig:
            return OmegaConf.create(run.config)

        def row_fun(r):
            return cfg(r).generative_rendering.do_post_attn_injection

        def col_fun(r):
            return cfg(r).animation.artifact_tag

        # group runs into grid
        runs_grid = group_into_array(
            runs,
            dim_key_fns=[row_fun, col_fun],
        )

        def get_video(run):
            video_artifact = wbu.first_logged_artifact_of_type(run, "video")
            video_artifact = VideoArtifact.from_wandb_artifact(video_artifact)
            video_frames = video_artifact.get_moviepy_clip()
            return video_frames

        videos = np.vectorize(get_video)(runs_grid)

        if show_depth_videos:
            # for each column get depth-video
            depth_videos = []
            for run in runs_grid[0]:
                config = cfg(run)
                n_frames = config.animation.n_frames

                # get animation
                animation = wbu.first_used_artifact_of_type(run, "animation")
                animation = AnimationArtifact.from_wandb_artifact(animation)
                frame_indices = animation.frame_indices(n_frames)
                cams, meshes = animation.load_frames(frame_indices)

                # get depth maps
                depth_maps = render_depth_map(meshes, cams)
                depth_video = pil_frames_to_clip(depth_maps, fps=10)
                depth_videos.append(depth_video)

            videos.insert(0, depth_videos)

        comparison_vid = video_grid(videos)
        return comparison_vid
