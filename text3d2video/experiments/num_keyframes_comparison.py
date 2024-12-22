from typing import List

from hydra.compose import compose
from hydra.initialize import initialize
from omegaconf import OmegaConf

import text3d2video.wandb_util as wbu
from scripts import run_generative_rendering
from scripts.run_generative_rendering import RunGenerativeRenderingConfig
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.evaluation.video_comparison import video_grid
from text3d2video.experiment_util import WandbExperiment


class NKeyframesExperiment(WandbExperiment):
    experiment_name = "experiment_1"
    n_inference_steps: int
    kf_vals: List[int]
    animation_artifact_tag: str = "mixamo-human_rotation:latest"

    def __init__(self, kf_vals: List[int], n_inference_steps=10):
        self.n_inference_steps = n_inference_steps
        self.kf_vals = kf_vals
        self.run_fn = run_generative_rendering.run

    def run_configs(self):
        configs = []

        for n_kf in self.kf_vals:
            with initialize(version_base=None, config_path="config"):
                cfg: RunGenerativeRenderingConfig = compose(
                    config_name="generative_rendering",
                )
                cfg.generative_rendering.num_keyframes = n_kf
                cfg.animation.artifact_tag = self.animation_artifact_tag
                cfg.generative_rendering.num_inference_steps = self.n_inference_steps
                configs.append(cfg)

        return configs

    @classmethod
    def plot_results(self, group: str, labels=True):
        runs = self.get_runs_in_group(group)

        # sort by number of keyframes
        def kf_fun(run):
            cfg: RunGenerativeRenderingConfig = OmegaConf.create(run.config)
            return cfg.generative_rendering.num_keyframes

        runs = sorted(runs, key=kf_fun)

        # get videos
        vids = []
        for r in runs:
            vid = wbu.first_logged_artifact_of_type(
                r, VideoArtifact.wandb_artifact_type
            )
            vid = VideoArtifact.from_wandb_artifact(vid)
            vids.append(vid.get_moviepy_clip())

        x_labels = [f"KF: {kf_fun(r)}" for r in runs]

        if not labels:
            x_labels = None
        return video_grid([vids], x_labels=x_labels)
