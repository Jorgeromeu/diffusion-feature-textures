from dataclasses import dataclass
from enum import Enum

from text3d2video.artifacts.gr_data import GrSaveConfig


@dataclass
class RunConfig:
    job_type: str
    wandb: bool
    instant_exit: bool
    download_artifacts: bool
    tags: list[str]


class NoiseInitializationMethod(Enum):
    RANDOM = "RANDOM"
    FIXED = "FIXED"
    UV = "UV"


@dataclass
class NoiseInitializationConfig:
    method: NoiseInitializationMethod
    uv_texture_res = -1


@dataclass
class AnimationConfig:
    n_frames: int
    artifact_tag: str


# pylint: disable=too-many-instance-attributes
@dataclass
class GenerativeRenderingConfig:
    seed: int
    resolution: int
    do_pre_attn_injection: bool
    do_post_attn_injection: bool
    feature_blend_alpha: float
    attend_to_self_kv: bool
    mean_features_weight: float
    chunk_size: int
    num_keyframes: int
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    module_paths: list[str]


@dataclass
class RunGenerativeRenderingConfig:
    prompt: str
    animation: AnimationConfig
    out_artifact: str
    run: RunConfig
    save_tensors: GrSaveConfig
    noise_initialization: NoiseInitializationConfig
    generative_rendering: GenerativeRenderingConfig
