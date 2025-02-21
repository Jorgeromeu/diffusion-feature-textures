from dataclasses import dataclass

import numpy as np
from hydra.compose import compose
from hydra.initialize import initialize
from omegaconf import OmegaConf

import scripts.run_generative_rendering
import scripts.run_reposable_diffusion
from scripts.run_generative_rendering import ModelConfig
from scripts.run_reposable_diffusion import RunReposableDiffusionConfig
from text3d2video.artifacts.anim_artifact import AnimationConfig
from text3d2video.artifacts.gr_data import GrSaveConfig
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.noise_initialization import UVNoiseInitializer
from text3d2video.pipelines.reposable_diffusion_pipeline import ReposableDiffusionConfig
from text3d2video.utilities.video_comparison import (
    group_into_array,
    video_grid,
)
from text3d2video.utilities.video_util import extend_clip_to_match_duration
from wandb_util.experiment_util import (
    WandbExperiment,
    object_to_instantiate_config,
)
from wandb_util.wandb_util import RunConfig

"""
Example experiment which runs generative rendering on a set of scenes and prompts
"""


@dataclass
class SceneConfig:
    prompt: str
    src_animation: AnimationConfig
    tgt_animation: AnimationConfig


@dataclass
class ReposableDiffusionAblationExperimentConfig:
    run: RunConfig
    scenes: list[SceneConfig]
    save_tensors: GrSaveConfig
    reposable_diffusion: ReposableDiffusionConfig
    model: ModelConfig


@dataclass(frozen=True, order=True)
class InjectionConfig:
    index: int
    name: str
    attend_to_self: bool
    pre_attn_injection: bool
    post_attn_injection: bool
    aggregate_queries: bool


class ReposableDiffusionAblationExperiment(WandbExperiment):
    injection_configs = [
        InjectionConfig(
            name="No Injection",
            attend_to_self=False,
            pre_attn_injection=False,
            post_attn_injection=False,
            aggregate_queries=False,
            index=0,
        ),
        InjectionConfig(
            name="Attend to src",
            attend_to_self=False,
            pre_attn_injection=True,
            post_attn_injection=False,
            aggregate_queries=False,
            index=1,
        ),
        InjectionConfig(
            name="Attend to src + Self",
            attend_to_self=True,
            pre_attn_injection=True,
            post_attn_injection=False,
            aggregate_queries=False,
            index=2,
        ),
        InjectionConfig(
            name="Post Attn",
            attend_to_self=False,
            pre_attn_injection=False,
            post_attn_injection=True,
            aggregate_queries=False,
            index=3,
        ),
        InjectionConfig(
            name="Post Attn + Attend to src",
            attend_to_self=False,
            pre_attn_injection=True,
            post_attn_injection=True,
            aggregate_queries=False,
            index=4,
        ),
        InjectionConfig(
            name="Post Attn + Attend to src+self",
            attend_to_self=True,
            pre_attn_injection=True,
            post_attn_injection=True,
            aggregate_queries=False,
            index=5,
        ),
        InjectionConfig(
            name="Qry Injection + Attend to src",
            attend_to_self=True,
            pre_attn_injection=True,
            post_attn_injection=True,
            aggregate_queries=True,
            index=6,
        ),
    ]

    def __init__(self):
        self.run_fn = scripts.run_reposable_diffusion.run
        self.experiment_name = "reposable_diffusion_ablation"

    def run_configs(self):
        configs = []

        with initialize(version_base=None, config_path="../../config"):
            config: ReposableDiffusionAblationExperimentConfig = compose(
                config_name=self.experiment_name
            )

        for scene in config.scenes:
            for injection_config in self.injection_configs:
                gr_config: ReposableDiffusionConfig = config.reposable_diffusion.copy()

                # set ablation config
                gr_config.do_pre_attn_injection = injection_config.pre_attn_injection
                gr_config.do_post_attn_injection = injection_config.post_attn_injection
                gr_config.attend_to_self_kv = injection_config.attend_to_self
                gr_config.aggregate_queries = injection_config.aggregate_queries

                run_config: RunConfig = config.run.copy()
                run_config.name = injection_config.name

                cfg = RunReposableDiffusionConfig(
                    run=config.run,
                    video_artifact="video",
                    aggr_artifact="aggr",
                    prompt=scene.prompt,
                    target_frames=scene.tgt_animation,
                    source_frames=scene.src_animation,
                    reposable_diffusion=gr_config,
                    noise_initialization=object_to_instantiate_config(
                        UVNoiseInitializer()
                    ),
                    save_tensors=config.save_tensors,
                    model=config.model,
                )

                cfg = OmegaConf.structured(cfg)
                configs.append(cfg)

        return configs

    def video_comparison(self, runs=None, clip_metrics=None, with_text=False):
        if runs is None:
            runs = self.get_logged_runs()

        def run_cfg(run) -> RunReposableDiffusionConfig:
            return OmegaConf.create(run.config)

        def scene_key(run):
            cfg = run_cfg(run)
            prompt = cfg.prompt
            src_anim = cfg.source_frames
            tgt_anim = cfg.target_frames
            return f"{prompt}-{src_anim}-{tgt_anim}"

        def injection_config_key(run):
            cfg = run_cfg(run)

            injection_config = None
            for config in self.injection_configs:
                if (
                    config.pre_attn_injection
                    == cfg.reposable_diffusion.do_pre_attn_injection
                    and config.post_attn_injection
                    == cfg.reposable_diffusion.do_post_attn_injection
                    and config.attend_to_self
                    == cfg.reposable_diffusion.attend_to_self_kv
                    and config.aggregate_queries
                    == cfg.reposable_diffusion.aggregate_queries
                ):
                    injection_config = config

            return injection_config

        runs_grid, dim_keys = group_into_array(
            runs,
            dim_key_fns=[scene_key, injection_config_key],
            return_keys=True,
        )
        col_labels = [key.name for key in dim_keys[1]]

        def get_videos(run):
            artifacts = list(run.logged_artifacts())

            vid_art = None
            aggr_art = None
            for artifact in artifacts:
                if artifact.name.startswith("video"):
                    vid_art = artifact
                elif artifact.name.startswith("aggr"):
                    aggr_art = artifact

            vid_art = VideoArtifact.from_wandb_artifact(vid_art)
            video = vid_art.get_moviepy_clip()

            aggr_art = VideoArtifact.from_wandb_artifact(aggr_art)
            aggr_vid = aggr_art.get_moviepy_clip()

            return video, aggr_vid

        videos_grid, aggr_vids_grid = np.vectorize(get_videos)(runs_grid)
        aggr_vids = aggr_vids_grid[:, 0]

        # extend clips to match duration
        for i, video in enumerate(aggr_vids):
            aggr_vids[i] = extend_clip_to_match_duration(
                video, videos_grid[0, 0].duration
            )

        all_vids = np.concatenate([aggr_vids[:, np.newaxis], videos_grid], axis=1)

        labels = ["Source"] + col_labels
        if not with_text:
            labels = None

        comparison_grid = video_grid(
            all_vids, x_labels=labels, col_gap_indices=[0, 1, 3, 6]
        )
        return comparison_grid
