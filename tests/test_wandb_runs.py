from omegaconf import OmegaConf

import wandb_util.wandb_util as wbu
from scripts.wandb_runs.make_texture import MakeTextureConfig, make_texture
from scripts.wandb_runs.run_generative_rendering import (
    RunGenerativeRenderingConfig,
    run_generative_rendering,
)
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.pipelines.pipeline_utils import ModelConfig
from text3d2video.pipelines.texturing_pipeline import TexturingConfig


def test_run_gr():
    config = RunGenerativeRenderingConfig(
        "Cat",
        "mv_cat_statue:latest",
        GenerativeRenderingConfig(num_inference_steps=2),
        ModelConfig(),
    )
    config = OmegaConf.create(config)

    run_generative_rendering(config, wbu.RunConfig(wandb=False))


def test_make_texture():
    config = MakeTextureConfig(
        "Cat",
        "mv_cat_statue:latest",
        ModelConfig(),
        TexturingConfig(num_inference_steps=2),
    )
    config = OmegaConf.create(config)

    make_texture(config, wbu.RunConfig(wandb=False))
