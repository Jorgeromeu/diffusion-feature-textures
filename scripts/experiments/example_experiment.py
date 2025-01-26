from dataclasses import dataclass

from hydra.compose import compose
from hydra.initialize import initialize
from omegaconf import OmegaConf

import scripts.run_generative_rendering
import text3d2video.wandb_util as wbu
from scripts.run_generative_rendering import ModelConfig, RunGenerativeRenderingConfig
from text3d2video.artifacts.gr_data import GrSaveConfig
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.evaluation.video_comparison import group_and_sort, video_grid
from text3d2video.experiment_util import WandbExperiment, object_to_instantiate_config
from text3d2video.generative_rendering.configs import (
    AnimationConfig,
    GenerativeRenderingConfig,
    RunConfig,
)
from text3d2video.noise_initialization import RandomNoiseInitializer

"""
Example experiment which runs generative rendering on a set of scenes and prompts
"""


@dataclass
class ExampleExperimentConfig:
    model: ModelConfig
    run: RunConfig
    prompts: list[str]
    animations: list[AnimationConfig]
    save_tensors: GrSaveConfig
    generative_rendering: GenerativeRenderingConfig


class ExampleExperiment(WandbExperiment):
    def __init__(self):
        self.run_fn = scripts.run_generative_rendering.run
        self.experiment_name = "example_experiment"

    def run_configs(self):
        configs = []

        with initialize(version_base=None, config_path="../../config"):
            config: ExampleExperimentConfig = compose(config_name="example_experiment")

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

    @classmethod
    def video_comparison(cls, group: str = None):
        runs = cls.get_logged_runs(group)

        def run_cfg(run) -> RunGenerativeRenderingConfig:
            return OmegaConf.create(run.config)

        runs_grouped = group_and_sort(
            runs,
            group_fun=lambda r: run_cfg(r).prompt,
            sort_x_fun=lambda r: run_cfg(r).animation.artifact_tag,
        )

        videos_grid = []
        for row in runs_grouped:
            row_videos = []
            for run in row:
                video_artifact = wbu.first_logged_artifact_of_type(run, "video")
                video_artifact = VideoArtifact.from_wandb_artifact(video_artifact)
                row_videos.append(video_artifact.get_moviepy_clip())
            videos_grid.append(row_videos)

        comparison_vid = video_grid(videos_grid)
        return comparison_vid
