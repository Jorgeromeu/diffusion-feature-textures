from dataclasses import dataclass

from git import Optional


@dataclass
class RunConfig:
    wandb: bool  # if wandb is enabled
    instant_exit: bool  # if instant exit is enabled, do not run
    download_artifacts: bool  # if artifacts should be force-downloaded
    name: Optional[str]  # name of run in wandb
    job_type: Optional[str]  # job type in wandb
    group: Optional[str]  # group in wandb
    tags: list[str]  # tags for wandb run


@dataclass
class AnimationConfig:
    n_frames: int
    artifact_tag: str

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False


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
class ReposableDiffusionConfig:
    seed: int
    resolution: int
    do_pre_attn_injection: bool
    do_post_attn_injection: bool
    feature_blend_alpha: float
    attend_to_self_kv: bool
    mean_features_weight: float
    chunk_size: int
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    module_paths: list[str]
