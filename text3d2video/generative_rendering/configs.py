from dataclasses import dataclass


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


# pylint: disable=too-many-instance-attributes
@dataclass
class GenerativeRenderingConfig:
    res: int
    seed: int
    uv_texture_res: int
    do_uv_noise_init: bool
    do_pre_attn_injection: bool
    do_post_attn_injection: bool
    feature_blend_alpha: float
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
