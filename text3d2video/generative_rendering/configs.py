from dataclasses import dataclass

from git import Optional


@dataclass
class RunConfig:
    wandb: bool
    job_type: Optional[str]
    group: Optional[str]
    instant_exit: bool
    download_artifacts: bool
    tags: list[str]


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
