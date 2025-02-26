from dataclasses import dataclass

import numpy as np
from hydra.compose import compose
from hydra.initialize import initialize
from omegaconf import OmegaConf

import scripts.run_generative_rendering
import wandb_util.wandb_util as wbu
from scripts.run_generative_rendering import ModelConfig, RunGenerativeRenderingConfig
from text3d2video.artifacts.anim_artifact import AnimationConfig
from text3d2video.artifacts.gr_data import GrSaveConfig
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.noise_initialization import UVNoiseInitializer
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.utilities.video_comparison import (
    group_into_array,
    video_grid,
)
from wandb_util.experiment_util import (
    WandbExperiment,
    object_to_instantiate_config,
)

"""
Example experiment which runs generative rendering on a set of scenes and prompts
"""


@dataclass
class ExampleExperimentConfig:
    model: ModelConfig
    run: wbu.RunConfig
    prompts: list[str]
    animations: list[AnimationConfig]
    save_tensors: GrSaveConfig
    generative_rendering: GenerativeRenderingConfig


class ExampleExperiment(WandbExperiment):
    def __init__(self):
        self.run_fn = scripts.run_generative_rendering.main
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
                        UVNoiseInitializer()
                    ),
                    model=config.model,
                )

                cfg = OmegaConf.structured(cfg)

                configs.append(cfg)

        return configs

    def video_comparison(self):
        runs = self.get_logged_runs()

        def run_cfg(run) -> RunGenerativeRenderingConfig:
            return OmegaConf.create(run.config)

        def row_key(run):
            return run_cfg(run).prompt

        def col_key(run):
            return run_cfg(run).animation.artifact_tag

        runs_grouped = group_into_array(runs, dim_key_fns=[row_key, col_key])

        def get_video(run):
            video_artifact = wbu.first_logged_artifact_of_type(run, "video")
            video_artifact = VideoArtifact.from_wandb_artifact(video_artifact)
            return video_artifact.get_moviepy_clip()

        videos_grid = np.vectorize(get_video)(runs_grouped)

        comparison_vid = video_grid(videos_grid)
        return comparison_vid
