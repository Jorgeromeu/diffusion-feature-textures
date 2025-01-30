from dataclasses import dataclass

import numpy as np
from hydra.compose import compose
from hydra.initialize import initialize
from omegaconf import OmegaConf

import scripts.run_generative_rendering
import scripts.run_reposable_diffusion
from scripts.run_generative_rendering import ModelConfig
from scripts.run_reposable_diffusion import RunReposableDiffusionConfig
from text3d2video.artifacts.gr_data import GrSaveConfig
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.evaluation.video_comparison import (
    VideoLabel,
    add_label_to_clip,
    group_and_sort,
    video_grid,
)
from text3d2video.experiment_util import WandbExperiment, object_to_instantiate_config
from text3d2video.generative_rendering.configs import (
    AnimationConfig,
    GenerativeRenderingConfig,
    RunConfig,
)
from text3d2video.ipython_utils import map_list_of_lists
from text3d2video.noise_initialization import UVNoiseInitializer
from text3d2video.video_util import extend_clip_to_match_duration

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
    generative_rendering: GenerativeRenderingConfig
    model: ModelConfig


@dataclass
class InjectionConfig:
    name: str
    attend_to_self: bool
    pre_attn_injection: bool
    post_attn_injection: bool
    index: int


class ReposableDiffusionAblationExperiment(WandbExperiment):
    injection_configs = [
        InjectionConfig(
            name="No Injection",
            attend_to_self=False,
            pre_attn_injection=False,
            post_attn_injection=False,
            index=0,
        ),
        InjectionConfig(
            name="Inject KV",
            attend_to_self=False,
            pre_attn_injection=True,
            post_attn_injection=False,
            index=1,
        ),
        InjectionConfig(
            name="Inject KV + Attend to Self",
            attend_to_self=True,
            pre_attn_injection=True,
            post_attn_injection=False,
            index=2,
        ),
        InjectionConfig(
            name="Inject Post",
            attend_to_self=False,
            pre_attn_injection=False,
            post_attn_injection=True,
            index=3,
        ),
        InjectionConfig(
            name="Inject KV + Post Attn",
            attend_to_self=False,
            pre_attn_injection=True,
            post_attn_injection=True,
            index=4,
        ),
        InjectionConfig(
            name="Inject KV + Post attn + Attend to Self",
            attend_to_self=True,
            pre_attn_injection=True,
            post_attn_injection=True,
            index=5,
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
                gr_config = config.generative_rendering.copy()

                # set ablation config
                gr_config.do_pre_attn_injection = injection_config.pre_attn_injection
                gr_config.do_post_attn_injection = injection_config.post_attn_injection
                gr_config.attend_to_self_kv = injection_config.attend_to_self

                cfg = RunReposableDiffusionConfig(
                    run=config.run,
                    video_artifact="video",
                    aggr_artifact="aggr",
                    prompt=scene.prompt,
                    target_frames=scene.tgt_animation,
                    source_frames=scene.src_animation,
                    generative_rendering=gr_config,
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
            src_anim = cfg.source_frames.artifact_tag
            tgt_anim = cfg.target_frames.artifact_tag
            return f"{prompt}-{src_anim}-{tgt_anim}"

        def injection_config_key(run):
            cfg = run_cfg(run)

            injection_config = None
            for config in self.injection_configs:
                if (
                    config.pre_attn_injection
                    == cfg.generative_rendering.do_pre_attn_injection
                    and config.post_attn_injection
                    == cfg.generative_rendering.do_post_attn_injection
                    and config.attend_to_self
                    == cfg.generative_rendering.attend_to_self_kv
                ):
                    injection_config = config

            return injection_config

        runs_grouped = group_and_sort(
            runs,
            group_fun=scene_key,
            sort_x_fun=lambda r: injection_config_key(r).index,
        )

        def get_output_video(run):
            artifacts = list(run.logged_artifacts())

            vid_art = None
            for artifact in artifacts:
                if artifact.name.startswith("video"):
                    vid_art = artifact

            vid_art = VideoArtifact.from_wandb_artifact(vid_art)
            video = vid_art.get_moviepy_clip()

            if clip_metrics is not None:
                prompt_fidelity = clip_metrics.prompt_fidelity(vid_art)
                frame_consistency = clip_metrics.frame_consistency(vid_art)
                label_txt = f"PF: {prompt_fidelity:.4f}\nFC: {frame_consistency:.4f}"
                label = VideoLabel(content=label_txt)
                if with_text:
                    video = add_label_to_clip(video, label, position=("left", "bottom"))

            return video

        videos_grouped = map_list_of_lists(runs_grouped, get_output_video)
        videos_grouped = np.array(videos_grouped, dtype=object)

        # get all aggr vids
        aggr_vids = []
        for row in runs_grouped:
            r = row[0]
            artifacts = list(r.logged_artifacts())
            aggr_artifact = None
            for artifact in artifacts:
                if artifact.name.startswith("aggr"):
                    aggr_artifact = artifact

            aggr_artifact = VideoArtifact.from_wandb_artifact(aggr_artifact)
            aggr_vid = aggr_artifact.get_moviepy_clip()

            aggr_vid = extend_clip_to_match_duration(
                aggr_vid, videos_grouped[0][0].duration
            )
            aggr_vids.append(aggr_vid)

        aggr_vids = np.array(aggr_vids, dtype=object)
        all_vids = np.concatenate([aggr_vids[:, np.newaxis], videos_grouped], axis=1)

        row = runs_grouped[0]
        video_labels = [injection_config_key(r).name for r in row]

        labels = ["Aggregation Poses"] + video_labels
        if not with_text:
            labels = None
        comparison_grid = video_grid(all_vids, x_labels=labels)
        return comparison_grid
