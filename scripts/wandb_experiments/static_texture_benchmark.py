from typing import List

from attr import dataclass
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

import wandb_util.wandb_util as wbu
from text3d2video.utilities.omegaconf_util import omegaconf_from_dotdict


@dataclass
class Scene:
    animation_tag: str
    prompts: List[str]


@dataclass
class Method:
    name: str
    base_config: DictConfig
    fun_path: str


@dataclass
class StaticTextureBenchmarkConfig:
    scenes: List[Scene]
    methods: List[Method]


def texturing_benchmark(config: StaticTextureBenchmarkConfig):
    spec = []
    for method in config.methods:
        base_config = method.base_config
        run_fun = get_method(method.fun_path)

        for scene in config.scenes:
            for prompt in scene.prompts:
                scene_overrides = omegaconf_from_dotdict(
                    {"prompt": prompt, "animation_tag": scene.animation_tag}
                )

                overriden = OmegaConf.merge(base_config, scene_overrides)

                run_spec = wbu.RunSpec(
                    f"{method.name}_{scene.animation_tag}_{prompt}",
                    run_fun,
                    overriden,
                )

                spec.append(run_spec)

    return spec
