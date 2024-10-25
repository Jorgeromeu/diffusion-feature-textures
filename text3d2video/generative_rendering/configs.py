from dataclasses import dataclass
from enum import Enum


@dataclass
class RunConfig:
    job_type: str
    wandb: bool
    instant_exit: bool
    download_artifacts: bool
    tags: list[str]


@dataclass
class RerunConfig:
    enabled: bool
    module_paths: list[str]
    n_frames: int


@dataclass
class SaveConfig:
    enabled: bool
    save_latents: bool
    n_frames: int
    out_artifact: str


class NoiseInitializationMethod(Enum):
    RANDOM: str = "RANDOM"
    FIXED: str = "FIXED"
    UV: str = "UV"


@dataclass
class NoiseInitializationConfig:
    method: NoiseInitializationMethod
    uv_texture_res = -1


# pylint: disable=too-many-instance-attributes
@dataclass
class GenerativeRenderingConfig:
    seed: int
    resolution: int
    noise_initialization: NoiseInitializationConfig
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
    out_artifact: str
    run: RunConfig
    rerun: RerunConfig
    save_tensors: SaveConfig
    generative_rendering: GenerativeRenderingConfig
