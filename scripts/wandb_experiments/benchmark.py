from typing import List

from attr import asdict, dataclass
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes

import wandb_util.wandb_util as wbu
from scripts.wandb_runs.make_texture import MakeTextureConfig, make_texture
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.clip_metrics import CLIPMetrics
from text3d2video.pipelines.pipeline_utils import ModelConfig
from text3d2video.pipelines.texturing_pipeline import TexturingConfig
from text3d2video.utilities.omegaconf_util import omegaconf_from_dotdict
from text3d2video.utilities.video_comparison import video_grid
from text3d2video.utilities.video_util import pil_frames_to_clip
from text3d2video.uv_consistency_metric import mean_uv_mse
from wandb.apis.public import Run


@dataclass
class GeometryAndPrompts:
    animation_tag: str
    texturing_tag: str
    src_tag: str
    prompts: List[str]
    n_seeds: int

    def make_vid(self, with_col_titles=True):
        def anim_uv_vid(anim_tag: str, fps=15):
            anim = AnimationArtifact.from_wandb_artifact_tag(anim_tag)
            uvs = anim.read_anim_seq().render_rgb_uv_maps()
            return pil_frames_to_clip(uvs, fps=fps)

        vids = [
            anim_uv_vid(tag)
            for tag in [self.animation_tag, self.texturing_tag, self.src_tag]
        ]

        col_titles = ["Animation", "Texturing", "src"] if with_col_titles else None

        label = f'"{str(self.prompts)}" (n_seeds: {self.n_seeds})'

        return video_grid(
            [vids],
            padding_mode="slow_down",
            x_labels=col_titles,
            y_labels=[label],
        )

    def to_scenes(self):
        scenes = []
        for prompt in self.prompts:
            for i in range(self.n_seeds):
                scene = Scene(
                    self.animation_tag,
                    self.texturing_tag,
                    self.src_tag,
                    prompt,
                    i,
                )
                scenes.append(scene)
        return scenes


@dataclass
class Scene:
    animation_tag: str
    texturing_tag: str
    src_tag: str
    prompt: str
    seed: int

    def tabulate_row(self):
        return asdict(self)

    def make_vid(self, with_col_titles=True):
        def anim_uv_vid(anim_tag: str, fps=15):
            anim = AnimationArtifact.from_wandb_artifact_tag(anim_tag)
            uvs = anim.read_anim_seq().render_rgb_uv_maps()
            return pil_frames_to_clip(uvs, fps=fps)

        vids = [
            anim_uv_vid(tag)
            for tag in [self.animation_tag, self.texturing_tag, self.src_tag]
        ]

        col_titles = ["Animation", "Texturing", "src"] if with_col_titles else None

        label = f'"{self.prompt}" (seed: {self.seed})'

        return video_grid(
            [vids],
            padding_mode="slow_down",
            x_labels=col_titles,
            y_labels=[label],
        )


@dataclass
class Method:
    name: str
    fun_path: str
    base_config: DictConfig

    def tabulate_row(self):
        return {
            "name": self.name,
            "fun": self.fun_path.split(".")[-1],
            "config": OmegaConf.to_yaml(self.base_config),
        }


@dataclass
class BenchmarkConfig:
    scenes: List[Scene]
    methods: List[Method]


def texture_identifier(prompt: str, texture_tag: str):
    return wbu.make_valid_artifact_name(f"{texture_tag}_{prompt}")


def get_texture_runs(config: BenchmarkConfig):
    # Get set of all textures to generate
    texture_scenes = set()
    for s in config.scenes:
        texture_scenes.add((s.texturing_tag, s.prompt))

    make_texture_runs = dict()
    texgen_config = TexturingConfig()
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
            texture_id = texture_identifier(scene.prompt, scene.texturing_tag)

            scene_overrides = omegaconf_from_dotdict(
                {
                    "prompt": scene.prompt,
                    "animation_tag": scene.animation_tag,
                }
            )

            uses_texture = "texture_tag" in method.base_config
            if uses_texture:
                scene_overrides["texture_tag"] = f"{texture_id}:latest"

            overriden = OmegaConf.merge(method.base_config, scene_overrides)

            fun = get_method(method.fun_path)
            run_spec = wbu.RunSpec(method.name, fun, overriden)

            # conditionally add texture dependency
            if uses_texture:
                make_texture_spec = texture_runs_dict[texture_id]
                run_spec.depends_on.append(make_texture_spec)

            specs.append(run_spec)

    return make_texture_runs + specs


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
