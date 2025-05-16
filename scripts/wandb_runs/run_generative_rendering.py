from dataclasses import dataclass

import torch
from git import Optional

import wandb_util.wandb_util as wbu
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.artifacts.texture_artifact import TextureArtifact
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
    GrPipeline,
)
from text3d2video.pipelines.pipeline_utils import (
    ModelConfig,
    load_pipeline,
)
from text3d2video.utilities.video_util import pil_frames_to_clip


@dataclass
class RunGenerativeRenderingConfig:
    prompt: str
    animation_tag: str
    generative_rendering: GenerativeRenderingConfig
    model: ModelConfig
    src_anim_tag: Optional[str] = None
    texture_tag: Optional[str] = None
    start_noise_level: float = 0.0
    seed: int = 0
    kf_seed: Optional[int] = None  # if none use main seed
    out_artifact: str = "video"


@wbu.wandb_run("run_generative_rendering")
def run_generative_rendering(
    cfg: RunGenerativeRenderingConfig, run_config: wbu.RunConfig
):
    # disable gradients
    torch.set_grad_enabled(False)

    # read animation
    animation = AnimationArtifact.from_wandb_artifact_tag(
        cfg.animation_tag, download=True
    )
    anim = animation.read_anim_seq()

    # read source animation
    if cfg.src_anim_tag is not None:
        src_animation = AnimationArtifact.from_wandb_artifact_tag(
            cfg.src_anim_tag, download=True
        )
        src_anim = src_animation.read_anim_seq()
    else:
        src_anim = None

    # read texture
    if cfg.texture_tag is not None:
        texture = TextureArtifact.from_wandb_artifact_tag(
            cfg.texture_tag, download=True
        )
        texture = texture.read_texture()
    else:
        texture = None

    # load pipeline
    device = torch.device("cuda")
    pipe = load_pipeline(GrPipeline, cfg.model.sd_repo, cfg.model.controlnet_repo)

    # set seeds
    generator = torch.Generator(device=device).manual_seed(cfg.seed)
    kf_seed = cfg.kf_seed if cfg.kf_seed is not None else cfg.seed
    kf_generator = torch.Generator(device=device).manual_seed(kf_seed)

    # inference
    video_frames = pipe(
        cfg.prompt,
        anim,
        conf=cfg.generative_rendering,
        src_anim=src_anim,
        texture=texture,
        start_noise_level=cfg.start_noise_level,
        generator=generator,
        kf_generator=kf_generator,
    ).images

    # save video
    video_artifact = VideoArtifact.create_empty_artifact("video")
    video_artifact.write_frames(video_frames)

    # log video to run
    wbu.log_moviepy_clip("video", pil_frames_to_clip(video_frames), fps=10)

    # save video artifact
    video_artifact.log()
