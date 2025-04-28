from typing import List

from attr import dataclass
from omegaconf import OmegaConf

import wandb_util.wandb_util as wbu
from scripts.wandb_runs.make_texture import MakeTextureConfig, make_texture
from text3d2video.pipelines.pipeline_utils import ModelConfig
from text3d2video.pipelines.texgen_pipeline import TexGenConfig
from text3d2video.utilities.omegaconf_util import omegaconf_from_dotdict


@dataclass
class Scene:
    animation_tag: str
    prompts: List[str]


@dataclass
class StaticTextureBenchmarkConfig:
    scenes: List[Scene]


def texturing_benchmark(config: StaticTextureBenchmarkConfig):
    base_config = OmegaConf.structured(
        MakeTextureConfig(
            prompt="",
            animation_tag="",
            model=ModelConfig(),
            texgen=TexGenConfig(module_paths=[0]),
        )
    )

    methods = [("TexGen", base_config, make_texture)]

    spec = []

    for name, base_cfg, fun in methods:
        for scene in config.scenes:
            for prompt in scene.prompts:
                scene_overrides = omegaconf_from_dotdict(
                    {"prompt": prompt, "animation_tag": scene.animation_tag}
                )

                overriden = OmegaConf.merge(base_cfg, scene_overrides)

                run_spec = wbu.RunSpec(
                    f"{name}_{scene.animation_tag}_{prompt}",
                    fun,
                    overriden,
                )

                spec.append(run_spec)

    return spec
