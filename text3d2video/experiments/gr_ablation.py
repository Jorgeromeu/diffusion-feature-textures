import itertools

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
from text3d2video.noise_initialization import (
    FixedNoiseInitializer,
    RandomNoiseInitializer,
    UVNoiseInitializer,
)


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

                # cfg.noise_initialization = OmegaConf.create()

                if uv_noise:
                    noise_initializer = UVNoiseInitializer()
                else:
                    noise_initializer = RandomNoiseInitializer()

                cfg.noise_initialization = object_to_instantiate_config(
                    noise_initializer
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
