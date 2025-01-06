from hydra.compose import compose
from hydra.initialize import initialize
from omegaconf import OmegaConf

from scripts import run_generative_rendering
from scripts.run_generative_rendering import RunGenerativeRenderingConfig
from text3d2video.evaluation.video_comparison import (
    group_and_sort,
    runs_grid,
)
from text3d2video.experiment_util import WandbExperiment, object_to_instantiate_config
from text3d2video.noise_initialization import RandomNoiseInitializer


class BadPosesExperiment(WandbExperiment):
    experiment_name = "bad_poses"

    def __init__(self, n_inference_steps=10, scenes=[]):
        self.n_inference_steps = n_inference_steps
        self.run_fn = run_generative_rendering.run
        self.scenes = scenes

    def run_configs(self):
        configs = []

        base_gr: RunGenerativeRenderingConfig = None
        with initialize(version_base=None, config_path="../../config"):
            base_gr = compose(config_name="generative_rendering")
            base_gr.generative_rendering.num_inference_steps = self.n_inference_steps

        base_per_frame: RunGenerativeRenderingConfig = base_gr.copy()
        base_per_frame.generative_rendering.do_post_attn_injection = False
        base_per_frame.generative_rendering.do_pre_attn_injection = False
        base_per_frame.noise_initialization = object_to_instantiate_config(
            RandomNoiseInitializer()
        )

        for scene in self.scenes:
            gr: RunGenerativeRenderingConfig = base_gr.copy()
            per_frame: RunGenerativeRenderingConfig = base_per_frame.copy()
            gr.animation.artifact_tag = scene
            per_frame.animation.artifact_tag = scene
            configs.append(gr)
            configs.append(per_frame)

        return configs

    @classmethod
    def plot_results(self, group: str, labels=True):
        def run_config(run) -> RunGenerativeRenderingConfig:
            return OmegaConf.create(run.config)

        # read runs
        runs = self.get_runs_in_group(group)

        def group_fun(run):
            cfg = run_config(run)
            return cfg.noise_initialization._target_

        # sort rows by noise initialization method
        def horizontal_sort(run):
            cfg = run_config(run)
            pre_attn = cfg.generative_rendering.do_pre_attn_injection
            post_attn = cfg.generative_rendering.do_post_attn_injection
            return (pre_attn, post_attn)

        # group by noise initialization method
        runs = group_and_sort(runs, group_fun=group_fun, sort_x_fun=horizontal_sort)

        def col_label_fun(run):
            cfg = run_config(run)
            pre_attn = cfg.generative_rendering.do_pre_attn_injection
            post_attn = cfg.generative_rendering.do_post_attn_injection
            return f"Pre: {pre_attn}, Post: {post_attn}"

        def row_label_fun(run):
            return "UV"

        return runs_grid(
            runs, x_label_fun=col_label_fun, y_label_fun=row_label_fun, labels=labels
        )
