from typing import List

import numpy as np
import torch
from attr import dataclass
from omegaconf import OmegaConf
from PIL.Image import Image

import wandb_util.wandb_util as wbu
from scripts.wandb_runs.run_generative_rendering import (
    RunGenerativeRendering,
    RunGenerativeRenderingConfig,
)
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.pipelines.pipeline_utils import ModelConfig
from text3d2video.utilities.video_comparison import video_grid
from text3d2video.utilities.video_util import pil_frames_to_clip


@dataclass
class LoggedData:
    frames: List[Image]
    config: RunGenerativeRendering


class GrComparison(wbu.Experiment):
    experiment_name = "gr_comparison"

    def specification(self):
        # overrides

        weights = list(np.linspace(0, 1, 5))

        overrides = [{"generative_rendering.feature_blend_alpha": f} for f in weights]

        # Base Config
        prompt = "Stormtrooper"
        anim_tag = "ymca_20:latest"
        seed = 0

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

        base_gr_config = GenerativeRenderingConfig(
            do_pre_attn_injection=True,
            do_post_attn_injection=True,
            module_paths=decoder_paths,
        )

        base_config = RunGenerativeRenderingConfig(
            prompt=prompt,
            animation_tag=anim_tag,
            generative_rendering=base_gr_config,
            model=ModelConfig(),
            seed=seed,
            kf_seed=0,
        )
        base_config = OmegaConf.structured(base_config)

        override_dictconfigs = [wbu.omegaconf_create_nested(o) for o in overrides]
        override_configs = [
            OmegaConf.merge(base_config, o) for o in override_dictconfigs
        ]

        runs = [
            wbu.RunSpecification(f"o_{i}", RunGenerativeRendering(), o)
            for i, o in enumerate(override_configs)
        ]

        return runs

    def get_data(self):
        runs = self.get_logged_runs()

        def get_frames(run):
            video = wbu.logged_artifacts(run, type="video")[0]
            video = VideoArtifact.from_wandb_artifact(video)
            return video.read_frames()

        return [
            LoggedData(get_frames(run), OmegaConf.create(run.config)) for run in runs
        ]

    def comparison_vid(self, data=None):
        if data is None:
            data = self.get_data()

        videos = [pil_frames_to_clip(v.frames) for v in data]

        def key(c: RunGenerativeRenderingConfig):
            return f"kf_seed: {c.kf_seed}"

        titles = [key(c.config) for c in data]

        comparison_vid = video_grid([videos], x_labels=titles)
        return comparison_vid
