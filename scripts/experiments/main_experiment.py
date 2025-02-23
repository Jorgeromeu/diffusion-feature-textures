import multiprocessing as mp
from dataclasses import dataclass
from typing import List

from omegaconf import DictConfig, OmegaConf

import wandb_util.wandb_util as wbu
from scripts.run_generative_rendering import (
    ModelConfig,
    RunGenerativeRendering,
    RunGenerativeRenderingConfig,
)
from scripts.run_reposable_diffusion_t2v import (
    RunReposableDiffusionT2V,
    RunReposableDiffusionT2VConfig,
)
from text3d2video.artifacts.anim_artifact import AnimationConfig
from text3d2video.noise_initialization import UVNoiseInitializer
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.pipelines.reposable_diffusion_pipeline import ReposableDiffusionConfig
from wandb_util.experiment_util import (
    Experiment,
    RunDescriptor,
    WandbRun,
    object_to_instantiate_config,
)

"""
Example experiment which runs generative rendering on a set of scenes and prompts
"""


@dataclass
class Scene:
    prompt: str
    artifact_tag: str
    n_frames: int


@dataclass
class MainExperimentConfig:
    model: ModelConfig
    generative_rendering: GenerativeRenderingConfig
    reposable_diffusion: ReposableDiffusionConfig
    scenes: list[Scene]


class MainExperiment(Experiment):
    experiment_name = "main_experiment"
    config: MainExperimentConfig

    def __init__(self, config: DictConfig):
        self.config = config

    def specification(self) -> List[RunDescriptor]:
        run_config = wbu.RunConfig(
            wandb=True,
            instant_exit=False,
            download_artifacts=False,
            name=None,
            group=None,
            tags=[],
        )

        noise_initialization = object_to_instantiate_config(UVNoiseInitializer())

        runs = []

        for scene in self.config.scenes:
            animation = AnimationConfig(
                n_frames=scene.n_frames,
                artifact_tag=scene.artifact_tag,
            )

            gr_config = RunGenerativeRenderingConfig(
                run=run_config,
                prompt=scene.prompt,
                animation=animation,
                generative_rendering=self.config.generative_rendering,
                model=self.config.model,
                noise_initialization=noise_initialization,
                seed=0,
            )

            rd_config = RunReposableDiffusionT2VConfig(
                run=run_config,
                prompt=scene.prompt,
                animation=animation,
                reposable_diffusion=self.config.reposable_diffusion,
                model=self.config.model,
                noise_initialization=noise_initialization,
                num_views=10,
                seed=0,
            )

            runs.append(
                RunDescriptor(OmegaConf.structured(gr_config), RunGenerativeRendering())
            )

            runs.append(
                RunDescriptor(
                    OmegaConf.structured(rd_config), RunReposableDiffusionT2V()
                )
            )

        return runs
