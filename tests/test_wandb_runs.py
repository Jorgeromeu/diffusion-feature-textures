from omegaconf import OmegaConf

import wandb_util.wandb_util as wbu
from scripts.wandb_runs.make_texture import MakeTextureConfig, make_texture
from scripts.wandb_runs.render_noise_gr import RenderNoiseGrConfig, render_noise_gr
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
    decoder_paths = [
        "mid_block.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1",
    ]
    config = RunGenerativeRenderingConfig(
        "Cat",
        "mv_cat_statue:latest",
        GenerativeRenderingConfig(decoder_paths, num_inference_steps=2),
        ModelConfig(),
    )
    config = OmegaConf.create(config)

    run_generative_rendering(config, wbu.RunConfig(wandb=False))


def test_make_texture():
    decoder_paths = [
        "mid_block.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1",
    ]
    config = MakeTextureConfig(
        "Cat",
        "mv_cat_statue:latest",
        ModelConfig(),
        TexturingConfig(module_paths=decoder_paths, num_inference_steps=2),
    )
    config = OmegaConf.create(config)

    make_texture(config, wbu.RunConfig(wandb=False))


def test_render_noise_gr():
    config = RenderNoiseGrConfig(
        prompt="Cat",
        animation_tag="mv_cat_statue:latest",
        texture_tag="cat_statue_mvlatest_SilverCatStatue:latest",
        generative_rendering=GenerativeRenderingConfig(
            module_paths=[
                "mid_block.attentions.0.transformer_blocks.0.attn1",
                "up_blocks.1.attentions.0.transformer_blocks.0.attn1",
                "up_blocks.1.attentions.1.transformer_blocks.0.attn1",
                "up_blocks.1.attentions.2.transformer_blocks.0.attn1",
                "up_blocks.2.attentions.0.transformer_blocks.0.attn1",
                "up_blocks.2.attentions.1.transformer_blocks.0.attn1",
                "up_blocks.2.attentions.2.transformer_blocks.0.attn1",
                "up_blocks.3.attentions.0.transformer_blocks.0.attn1",
                "up_blocks.3.attentions.1.transformer_blocks.0.attn1",
                "up_blocks.3.attentions.2.transformer_blocks.0.attn1",
            ],
            num_inference_steps=2,
        ),
        model=ModelConfig(),
        start_noise_level=0.0,
    )
    config = OmegaConf.create(config)
    render_noise_gr(config, wbu.RunConfig(wandb=False))
