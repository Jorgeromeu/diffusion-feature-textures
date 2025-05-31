from typing import List

from attr import asdict, dataclass
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

import wandb_util.wandb_util as wbu
from scripts.wandb_runs.texgen_extr import TexGenExtrConfig, run_texgen_extr
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.pipelines.pipeline_utils import ModelConfig
from text3d2video.pipelines.texturing_pipeline import TexGenConfig
from text3d2video.utilities.video_comparison import video_grid
from text3d2video.utilities.video_util import pil_frames_to_clip
from wandb.apis.public import Run


@dataclass
class GeometryAndPrompts:
    animation_tag: str
    texturing_tag: str
    prompts: List[str]
    n_seeds: int

    def to_scenes(self):
        scenes = []
        for prompt in self.prompts:
            for i in range(self.n_seeds):
                scene = Scene(
                    self.animation_tag,
                    self.texturing_tag,
                    prompt,
                    i,
                )
                scenes.append(scene)
        return scenes


@dataclass
class Scene:
    animation_tag: str
    texturing_tag: str
    prompt: str
    seed: int

    def tabulate_row(self):
        return asdict(self)

    def make_vid(self, with_col_titles=True):
        def anim_uv_vid(anim_tag: str, fps=15):
            anim = AnimationArtifact.from_wandb_artifact_tag(anim_tag)
            uvs = anim.read_anim_seq().render_rgb_uv_maps()
            return pil_frames_to_clip(uvs, fps=fps)

        vids = {}
        vids["Anim"] = anim_uv_vid(self.animation_tag)
        vids["Tex"] = anim_uv_vid(self.texturing_tag)

        label = f'"{self.prompt}" (seed: {self.seed})'
        titles = list(vids.keys()) if with_col_titles else None
        vids = list(vids.values())

        return video_grid(
            [vids],
            padding_mode="slow_down",
            x_labels=titles,
            y_labels=[label],
        )


@dataclass
class Method:
    name: str
    fun_path: str
    base_config: DictConfig

    def tabulate_row(self, with_config: bool = True):
        row = {
            "name": self.name,
            "fun": self.fun_path.split(".")[-1],
        }
        if with_config:
            row["config"] = OmegaConf.to_yaml(self.base_config)

        return row


@dataclass
class BenchmarkConfig:
    scenes: List[Scene]
    methods: List[Method]


def texture_identifier(prompt: str, texture_tag: str, seed: int):
    return wbu.make_valid_artifact_name(f"{texture_tag}_{prompt}_{seed}")


def make_extr_runs(config: BenchmarkConfig):
    # set of unique tex_tag-prompt-seed tuples
    texture_scenes = set()
    for s in config.scenes:
        texture_scenes.add((s.texturing_tag, s.prompt, s.seed))

    # for each unique tex_tag-prompt-seed tuple, create a extraction run
    make_texture_runs = dict()
    texgen_config = TexGenConfig()
    for texturing_tag, texturing_prompt, seed in texture_scenes:
        tex_name = texture_identifier(texturing_prompt, texturing_tag, seed)
        make_texture_config = TexGenExtrConfig(
            texturing_prompt,
            texturing_tag,
            ModelConfig(),
            texgen_config,
            seed=seed,
            extr_out_art=tex_name,
        )
        make_texture_config = OmegaConf.structured(make_texture_config)

        spec = wbu.RunSpec(tex_name, run_texgen_extr, make_texture_config)
        make_texture_runs[tex_name] = spec

    # map extr_id to make_texture_run
    return make_texture_runs


def override_not_nones(base_config: DictConfig, overrides: DictConfig):
    overriden = base_config.copy()
    for key, value in overrides.items():
        use_field = hasattr(base_config, key) and getattr(base_config, key) is not None
        if use_field:
            overriden[key] = value
    return overriden


def benchmark(config: BenchmarkConfig):
    # Dict from texture name to make_texture run spec
    texture_runs_dict = make_extr_runs(config)
    make_texture_runs = list(texture_runs_dict.values())

    specs = []
    for method in config.methods:
        for scene in config.scenes:
            # get texture run corresponding to run
            texture_id = texture_identifier(
                scene.prompt, scene.texturing_tag, scene.seed
            )

            overrides = {
                "prompt": scene.prompt,
                "animation_tag": scene.animation_tag,
                "seed": scene.seed,
                "extr_tag": f"{texture_id}:latest",
                "src_anim_tag": scene.texturing_tag,
            }

            # override config with scene-specific values
            overriden = override_not_nones(method.base_config, overrides)

            # get method function
            fun = get_method(method.fun_path)
            run_spec = wbu.RunSpec(method.name, fun, overriden)

            # conditionally add texture dependency
            use_texture = hasattr(method.base_config, "extr_tag")
            if use_texture:
                make_texture_spec = texture_runs_dict[texture_id]
                run_spec.depends_on.append(make_texture_spec)

            specs.append(run_spec)

    return make_texture_runs + specs


def split_runs(runs: list[Run]):
    texture_runs = []
    video_gen_runs = []

    for run in runs:
        if run.job_type == "run_texgen_extr":
            texture_runs.append(run)
        else:
            video_gen_runs.append(run)

    return texture_runs, video_gen_runs
