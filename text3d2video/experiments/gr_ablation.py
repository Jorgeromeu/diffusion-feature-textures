import itertools

from httpx import post
from hydra.compose import compose
from hydra.initialize import initialize
from omegaconf import OmegaConf

import text3d2video.wandb_util as wbu
from scripts import run_generative_rendering
from scripts.run_generative_rendering import RunGenerativeRenderingConfig
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.evaluation.video_comparison import video_grid
from text3d2video.experiment_util import WandbExperiment
from text3d2video.generative_rendering.configs import (
    NoiseInitializationMethod,
)
from text3d2video.util import group_list_by


class GenRenderingAblation(WandbExperiment):
    experiment_name = "gr_ablation"
    animation_artifact_tag: str = "rumba:latest"
    prompt: str = "Deadpool Dancing"
    n_inference_steps: int

    def __init__(self, n_inference_steps=10):
        self.n_inference_steps = n_inference_steps
        self.run_fn = run_generative_rendering.run

    def run_configs(self):
        configs = []

        for uv_noise, pre_attn, post_attn in itertools.product(
            [True, False], [True, False], [True, False]
        ):
            with initialize(version_base=None, config_path="../../config"):
                cfg: RunGenerativeRenderingConfig = compose(
                    config_name="generative_rendering",
                )
                cfg.animation.artifact_tag = self.animation_artifact_tag
                cfg.generative_rendering.num_inference_steps = self.n_inference_steps
                cfg.prompt = self.prompt

                cfg.noise_initialization.method = (
                    NoiseInitializationMethod.UV
                    if uv_noise
                    else NoiseInitializationMethod.RANDOM
                )

                cfg.generative_rendering.do_pre_attn_injection = pre_attn
                cfg.generative_rendering.do_post_attn_injection = post_attn
                configs.append(cfg)

        return configs

    @classmethod
    def plot_results(self, group: str, labels=True):
        def run_config(run) -> RunGenerativeRenderingConfig:
            return OmegaConf.create(run.config)

        # read runs
        runs = self.get_runs_in_group(group)

        # group by noise initialization method
        runs = group_list_by(runs, lambda r: run_config(r).noise_initialization.method)

        def sort_fun(run):
            cfg = run_config(run)
            pre_attn = cfg.generative_rendering.do_pre_attn_injection
            post_attn = cfg.generative_rendering.do_post_attn_injection
            return (pre_attn, post_attn)

        # sort rows by noise initialization method
        runs = sorted(
            runs,
            key=lambda r: run_config(r[0]).noise_initialization.method,
            reverse=True,
        )

        # sort by pre and post attn injection
        for group in runs:
            group = sorted(group, key=sort_fun, reverse=True)

        run0 = runs[0][0]
        prompt = run_config(run0).prompt

        row_runs = [group[0] for group in runs]
        col_runs = runs[0]

        noise_names = {
            NoiseInitializationMethod.UV.value: "UV Noise",
            NoiseInitializationMethod.RANDOM.value: "Random Noise",
            NoiseInitializationMethod.FIXED.value: "Fixed Noise",
        }

        row_titles = [
            noise_names[run_config(r).noise_initialization.method] for r in row_runs
        ]

        col_titles = []
        for r in col_runs:
            cfg = run_config(r)
            do_pre = cfg.generative_rendering.do_pre_attn_injection
            do_post = cfg.generative_rendering.do_post_attn_injection
            col_titles.append(f"Pre-Attn: {do_pre}, Post-Attn: {do_post}")

        # get videos
        vids = []
        for group in runs:
            group_vids = []
            for r in group:
                vid = wbu.first_logged_artifact_of_type(
                    r, VideoArtifact.wandb_artifact_type
                )
                vid = VideoArtifact.from_wandb_artifact(vid)
                group_vids.append(vid.get_moviepy_clip())
            vids.append(group_vids)

        return video_grid(vids)
