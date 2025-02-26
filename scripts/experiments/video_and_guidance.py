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


class VideoAndGuidance(wbu.Experiment):
    experiment_name = "example_video"
