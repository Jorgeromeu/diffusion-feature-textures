from typing import List

from attr import dataclass
from omegaconf import DictConfig, OmegaConf
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes

import wandb_util.wandb_util as wbu
from scripts.wandb_runs.make_texture import MakeTextureConfig, make_texture
from scripts.wandb_runs.render_noise_gr import RenderNoiseGrConfig, render_noise_gr
from scripts.wandb_runs.run_generative_rendering import (
    RunGenerativeRenderingConfig,
    run_generative_rendering,
)
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.clip_metrics import CLIPMetrics
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.pipelines.pipeline_utils import ModelConfig
from text3d2video.pipelines.texgen_pipeline import TexGenConfig
from text3d2video.utilities.omegaconf_util import omegaconf_from_dotdict
from text3d2video.uv_consistency_metric import mean_uv_mse
from wandb.apis.public import Run


@dataclass
class Scene:
    animation_tag: str
    texturing_tag: str
    video_prompt: str
    texture_prompt: str


@dataclass
class Method:
    name: str
    fun_path: str
    base_config: DictConfig


@dataclass
class BenchmarkConfig:
    scenes: List[Scene]
    methods: List[Method]


def texture_identifier(prompt: str, texture_tag: str):
    return wbu.make_valid_artifact_name(f"{texture_tag}_{prompt}")


def get_texture_runs(config: BenchmarkConfig):
    decoder_paths = [
        "mid_block.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1",
    ]

    # Get set of all textures to generate
    texture_scenes = set()
    for s in config.scenes:
        texture_scenes.add((s.texturing_tag, s.texture_prompt))

    make_texture_runs = dict()
    texgen_config = TexGenConfig(decoder_paths)
    for texturing_tag, texturing_prompt in texture_scenes:
        tex_name = texture_identifier(texturing_prompt, texturing_tag)

        make_texture_config = MakeTextureConfig(
            texturing_prompt,
            texturing_tag,
            ModelConfig(),
            texgen_config,
            texture_out_art=tex_name,
        )
        make_texture_config = OmegaConf.structured(make_texture_config)

        spec = wbu.RunSpec(tex_name, make_texture, make_texture_config)
        make_texture_runs[tex_name] = spec

    return make_texture_runs


def benchmark(config: BenchmarkConfig):
    # Dict from texture name to make_texture run spec
    texture_runs_dict = get_texture_runs(config)
    make_texture_runs = list(texture_runs_dict.values())

    specs = []
    for method in config.methods:
        for scene in config.scenes:
            texture_id = texture_identifier(scene.texture_prompt, scene.texturing_tag)

            scene_overrides = omegaconf_from_dotdict(
                {
                    "prompt": scene.video_prompt,
                    "animation_tag": scene.animation_tag,
                }
            )

            uses_texture = "texture_tag" in method.base_config
            if uses_texture:
                scene_overrides["texture_tag"] = f"{texture_id}:latest"

            overriden = OmegaConf.merge(method.base_config, scene_overrides)
            run_spec = wbu.RunSpec(method.name, method, overriden)

            # conditionally add texture dependency
            if uses_texture:
                make_texture_spec = texture_runs_dict[texture_id]
                run_spec.depends_on.append(make_texture_spec)

            specs.append(run_spec)

    return make_texture_runs + specs


# Analysis


def split_runs(runs: list[Run]):
    texture_runs = []
    video_gen_runs = []

    for run in runs:
        if run.job_type == "make_texture":
            texture_runs.append(run)
        else:
            video_gen_runs.append(run)

    return texture_runs, video_gen_runs


@dataclass
class GrRunData:
    frames: list
    prompt: str
    cams: CamerasBase
    meshes: Meshes
    verts_uvs: list
    faces_uvs: list
    frame_consistency: float = None
    prompt_fidelity: float = None
    uv_mse: float = None

    @classmethod
    def from_run(cls, run: Run):
        # get video
        video = wbu.logged_artifacts(run, "video")[0]
        video = VideoArtifact.from_wandb_artifact(video)
        frames = video.read_frames()

        # get prompt
        prompt = OmegaConf.create(run.config).prompt

        # get anim
        anim = wbu.used_artifacts(run, "animation")[0]
        anim = AnimationArtifact.from_wandb_artifact(anim)
        cams, meshes = anim.load_frames()
        verts_uvs, faces_uvs = anim.uv_data()

        return cls(frames, prompt, cams, meshes, verts_uvs, faces_uvs)

    def compute_clip_metrics(self, model: CLIPMetrics):
        self.frame_consistency = model.frame_consistency(self.frames)
        self.prompt_fidelity = model.prompt_fidelity(self.frames, self.prompt)

    def compute_uv_mse(self):
        self.uv_mse = mean_uv_mse(
            self.frames, self.cams, self.meshes, self.verts_uvs, self.faces_uvs
        )
