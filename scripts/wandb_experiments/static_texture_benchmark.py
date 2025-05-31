from typing import List

from attr import dataclass
from omegaconf import DictConfig, OmegaConf

import wandb_util.wandb_util as wbu
from scripts.wandb_runs.texgen_extr import run_texgen_extr
from text3d2video.utilities.omegaconf_util import omegaconf_from_dotdict


@dataclass
class TexturingScene:
    animation_tag: str
    prompt: str
    seed: int

    def tabulate_row(self):
        return {
            "animation_tag": self.animation_tag,
            "prompt": self.prompt,
            "seed": self.seed,
        }


@dataclass
class TexturingGeometryAndPrompts:
    animation_tag: str
    prompts: List[str]
    n_seeds: int = 1

    def to_scenes(self):
        scenes = []
        for prompt in self.prompts:
            for i in range(self.n_seeds):
                scene = TexturingScene(
                    self.animation_tag,
                    prompt,
                    i,
                )
                scenes.append(scene)
        return scenes


@dataclass
class TexturingMethod:
    name: str
    base_config: DictConfig

    def tabulate_row(self, flat_config: bool = False):
        config_str = OmegaConf.to_yaml(self.base_config)
        if flat_config:
            config_str = str(self.base_config)

        return {
            "name": self.name,
            "base_config": config_str,
        }


@dataclass
class TexturingBenchmarkConfig:
    scenes: List[TexturingScene]
    methods: List[TexturingMethod]


def texturing_benchmark(config: TexturingBenchmarkConfig):
    spec = []
    for method in config.methods:
        base_config = method.base_config
        run_fun = run_texgen_extr

        for scene in config.scenes:
            scene_overrides = omegaconf_from_dotdict(
                {"prompt": scene.prompt, "animation_tag": scene.animation_tag}
            )

            overriden = OmegaConf.merge(base_config, scene_overrides)

            run_spec = wbu.RunSpec(
                f"{method.name}",
                run_fun,
                overriden,
            )

            spec.append(run_spec)

    return spec
