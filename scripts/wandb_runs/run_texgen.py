from dataclasses import dataclass

import torch

import wandb_util.wandb_util as wbu
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.pipelines.pipeline_utils import (
    ModelConfig,
    load_pipeline,
)
from text3d2video.pipelines.texgen_pipeline import TexGenConfig, TexGenPipeline
from text3d2video.utilities.video_util import pil_frames_to_clip


@dataclass
class RunTexGenConfig:
    prompt: str
    animation_tag: str
    seed: int
    model: ModelConfig
    texgen: TexGenConfig
    out_artifact: str = "texture_frames"


@wbu.wandb_run("run_texgen")
def run_texgen(cfg: RunTexGenConfig, run_config: wbu.RunConfig):
    # disable gradients
    torch.set_grad_enabled(False)

    # read animation
    animation = AnimationArtifact.from_wandb_artifact_tag(
        cfg.animation_tag, download=True
    )
    cam_frames, mesh_frames = animation.load_frames()
    verts_uvs, faces_uvs = animation.uv_data()

    # load pipeline
    device = torch.device("cuda")
    pipe = load_pipeline(TexGenPipeline, cfg.model.sd_repo, cfg.model.controlnet_repo)

    # set seed
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    images = pipe(
        cfg.prompt,
        mesh_frames,
        cam_frames,
        verts_uvs,
        faces_uvs,
        texgen_config=cfg.texgen,
        generator=generator,
    )

    # save video
    video_artifact = VideoArtifact.create_empty_artifact(cfg.out_artifact)
    video_artifact.write_frames(images)

    # log video to run
    wbu.log_moviepy_clip("video", pil_frames_to_clip(images), fps=10)

    # save video artifact
    video_artifact.log_if_enabled()
