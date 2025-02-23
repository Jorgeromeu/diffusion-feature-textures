from dataclasses import dataclass
from typing import List

import numpy as np
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
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.noise_initialization import UVNoiseInitializer
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.pipelines.reposable_diffusion_pipeline import ReposableDiffusionConfig
from text3d2video.utilities.video_comparison import video_grid
from text3d2video.utilities.video_util import extend_clip_to_match_duration
from wandb_util.experiment_util import (
    object_to_instantiate_config,
)


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


class MainExperiment(wbu.Experiment):
    experiment_name = "main_experiment"
    config: MainExperimentConfig

    def __init__(self, config: DictConfig):
        self.config = config

    def specification(self) -> List[wbu.RunDescriptor]:
        noise_initialization = object_to_instantiate_config(UVNoiseInitializer())

        runs = []

        run_config = wbu.RunConfig(
            wandb=True,
            instant_exit=False,
            download_artifacts=False,
            name=None,
            group=None,
            tags=[],
        )

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

            per_frame_gr_config: GenerativeRenderingConfig = (
                self.config.generative_rendering.copy()
            )
            per_frame_gr_config.do_pre_attn_injection = False
            per_frame_gr_config.do_post_attn_injection = False
            per_frame_config = RunGenerativeRenderingConfig(
                run=run_config,
                prompt=scene.prompt,
                animation=animation,
                generative_rendering=per_frame_gr_config,
                model=self.config.model,
                noise_initialization=noise_initialization,
                seed=0,
            )

            per_frame_run = wbu.RunDescriptor(
                RunGenerativeRendering(), OmegaConf.structured(per_frame_config)
            )

            gr_run = wbu.RunDescriptor(
                RunGenerativeRendering(), OmegaConf.structured(gr_config)
            )

            rd_run = wbu.RunDescriptor(
                RunReposableDiffusionT2V(), OmegaConf.structured(rd_config)
            )

            runs.append(gr_run)
            runs.append(rd_run)
            runs.append(per_frame_run)

        return runs

    def video_comparison(self):
        runs = self.get_logged_runs()

        for r in runs:
            if r.job_type == "run_generative_rendering":
                if OmegaConf.create(
                    r.config
                ).generative_rendering.do_pre_attn_injection:
                    gr_run = r
                else:
                    per_frame_run = r
            else:
                rd_run = r

        for art in rd_run.logged_artifacts():
            if art.name.startswith("video"):
                rd_video_art = art
            else:
                rd_aggr_art = art

        gr_video_art = wbu.first_logged_artifact_of_type(gr_run, "video")
        per_frame_video_art = wbu.first_logged_artifact_of_type(per_frame_run, "video")

        video_artifacts = [per_frame_video_art, gr_video_art, rd_video_art, rd_aggr_art]
        videos = [VideoArtifact.from_wandb_artifact(art) for art in video_artifacts]
        videos = [art.get_moviepy_clip() for art in videos]

        max_duration = max([v.duration for v in videos])
        videos = [extend_clip_to_match_duration(v, max_duration) for v in videos]

        videos_grid = np.array([videos])

        return video_grid(
            videos_grid,
            col_gap_indices=[0, 1],
            # x_labels=["Per Frame", "Generative Rendering", "Ours (Target)", "Ours (Source)"],
        )
