from omegaconf import OmegaConf

import wandb_util.wandb_util as wbu
from scripts.wandb_runs import texgen_extr
from scripts.wandb_runs.run_generative_rendering import (
    RunGenerativeRenderingConfig,
    run_generative_rendering,
)
from scripts.wandb_runs.run_grtex import RunGrTexConfig, run_gr_tex
from scripts.wandb_runs.texgen_extr import (
    TexGenExtrConfig,
    run_texgen_extr,
)
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.pipelines.pipeline_utils import ModelConfig
from text3d2video.pipelines.texturing_pipeline import TexGenConfig


def test_run_gr():
    config = RunGenerativeRenderingConfig(
        "Cat",
        "mv_cat_statue:latest",
        GenerativeRenderingConfig(num_inference_steps=2),
        ModelConfig(),
    )
    config = OmegaConf.create(config)

    run_generative_rendering(config, wbu.RunConfig(wandb=False))


def test_run_grtex():
    config = RunGrTexConfig(
        "Cat",
        "mv_cat_statue:latest",
        "extr_frames:v0",
        GenerativeRenderingConfig(num_inference_steps=15),
        ModelConfig(),
        multires_textures=True,
        start_noise_level=0.3,
    )
    config = OmegaConf.create(config)

    run_gr_tex(config, wbu.RunConfig(wandb=True))


def test_texgen_extr():
    config = TexGenExtrConfig(
        "Metalic Cat Statue",
        "mv_cat_statue:latest",
        ModelConfig(),
        TexGenConfig(num_inference_steps=2),
    )
    config = OmegaConf.create(config)

    run_texgen_extr(config, wbu.RunConfig(wandb=False))
