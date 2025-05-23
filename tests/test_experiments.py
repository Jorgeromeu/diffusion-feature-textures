from omegaconf import OmegaConf

from scripts.wandb_experiments.benchmark import (
    BenchmarkConfig,
    benchmark,
)
from scripts.wandb_experiments.benchmark import Method as BenchmarkMethod
from scripts.wandb_experiments.benchmark import Scene as BenchmarkScene
from scripts.wandb_experiments.static_texture_benchmark import (
    TexturingBenchmarkConfig,
    TexturingMethod,
    TexturingScene,
    texturing_benchmark,
)
from scripts.wandb_runs.make_texture import MakeTextureConfig, make_texture
from scripts.wandb_runs.run_generative_rendering import (
    RunGenerativeRenderingConfig,
    run_generative_rendering,
)
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.pipelines.pipeline_utils import ModelConfig
from text3d2video.pipelines.texturing_pipeline import TexturingConfig
from text3d2video.utilities.omegaconf_util import get_import_path


def test_benchmark():
    metal_cat = BenchmarkScene(
        "cat_statue_mv:latest",
        "cat_statue_mv:latest",
        "cat_statue_mv:latest",
        "Silver Cat Statue",
        0,
    )

    base_config = OmegaConf.structured(
        RunGenerativeRenderingConfig(
            prompt="",
            animation_tag="",
            generative_rendering=GenerativeRenderingConfig(),
            model=ModelConfig(),
        )
    )

    methods = [
        BenchmarkMethod("GR", get_import_path(run_generative_rendering), base_config)
    ]

    scenes = [metal_cat]

    config = BenchmarkConfig(scenes=scenes, methods=methods)
    config = OmegaConf.structured(config)
    benchmark(config)


def test_texturing_benchmark():
    base_config = OmegaConf.structured(
        MakeTextureConfig(
            prompt="",
            animation_tag="",
            texgen=TexturingConfig(),
            model=ModelConfig(),
        )
    )
    methods = [TexturingMethod("GR", base_config, get_import_path(make_texture))]
    scenes = [TexturingScene("mv_cat_statue:latest", ["Cat"])]

    config = TexturingBenchmarkConfig(scenes=scenes, methods=methods)
    config = OmegaConf.structured(config)
    spec = texturing_benchmark(config)

    assert len(spec) == len(methods) * len(scenes)
