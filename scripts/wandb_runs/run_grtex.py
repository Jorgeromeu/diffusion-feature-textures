from dataclasses import dataclass

import torch
from git import Optional

import wandb_util.wandb_util as wbu
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.artifacts.extr_frames_artifact import ExtrFramesArtifact
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.pipelines.new_gr_pipeline import GrPipelineNew
from text3d2video.pipelines.pipeline_utils import (
    ModelConfig,
    load_pipeline,
)
from text3d2video.utilities.video_util import pil_frames_to_clip


@dataclass
class RunGrTexConfig:
    prompt: str
    animation_tag: str
    extr_tag: str
    generative_rendering: GenerativeRenderingConfig
    model: ModelConfig
    multires_textures: bool = True
    start_noise_level: Optional[float] = None
    do_texture_noise_init: bool = True
    seed: int = 0
    out_artifact: str = "video"


@wbu.wandb_run("run_grtex")
def run_gr_tex(cfg: RunGrTexConfig, run_config: wbu.RunConfig):
    # disable gradients
    torch.set_grad_enabled(False)

    # read animation
    animation = AnimationArtifact.from_wandb_artifact_tag(
        cfg.animation_tag, download=True
    )
    tgt_seq = animation.read_anim_seq()

    # read extr_frames
    extr_frames = ExtrFramesArtifact.from_wandb_artifact_tag(
        cfg.extr_tag, download=True
    )

    # get src_seq
    extr_run = extr_frames.logged_by()
    anim_src = wbu.used_artifacts(extr_run, type="animation")[0]
    anim_src = AnimationArtifact.from_wandb_artifact(anim_src, download=True)
    src_seq = anim_src.read_anim_seq()

    # load pipeline
    device = torch.device("cuda")
    pipe: GrPipelineNew = load_pipeline(
        GrPipelineNew, cfg.model.sd_repo, cfg.model.controlnet_repo
    )

    # set seed
    generator = torch.Generator(device=device).manual_seed(cfg.seed)

    # inference
    video_frames = pipe(
        cfg.prompt,
        tgt_seq,
        src_seq,
        extr_frames.read_latents(),
        multires_textures=cfg.multires_textures,
        conf=cfg.generative_rendering,
        initial_texture=extr_frames.read_texture()
        if cfg.do_texture_noise_init
        else None,
        texture_noise_level=cfg.start_noise_level,
        generator=generator,
    ).images

    # log video to run
    wbu.log_moviepy_clip("video", pil_frames_to_clip(video_frames), fps=10)

    # save video
    video_artifact = VideoArtifact.create_empty_artifact("video")
    video_artifact.write_frames(video_frames)

    # save video artifact
    video_artifact.log()
