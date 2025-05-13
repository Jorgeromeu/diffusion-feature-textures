from dataclasses import dataclass

import torch

import wandb_util.wandb_util as wbu
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.artifacts.texture_artifact import TextureArtifact
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.rendering import render_texture
from text3d2video.utilities.video_util import pil_frames_to_clip


@dataclass
class RenderTextureConfig:
    animation_tag: str
    texture_tag: str
    prompt: str
    video_out: str = "video"


@wbu.wandb_run("render_texture")
def render_texture_to_anim(cfg: RenderTextureConfig, run_config: wbu.RunConfig):
    # disable gradients
    torch.set_grad_enabled(False)

    # read animation
    anim = AnimationArtifact.from_wandb_artifact_tag(cfg.animation_tag, download=True)
    seq = anim.read_anim_seq()

    # read texture
    tex_art = TextureArtifact.from_wandb_artifact_tag(cfg.texture_tag, download=True)
    texture = tex_art.read_texture()

    video_frames = render_texture(
        seq.meshes, seq.cams, texture, seq.verts_uvs, seq.faces_uvs, return_pil=True
    )

    # save video
    video_artifact = VideoArtifact.create_empty_artifact("video")
    video_artifact.write_frames(video_frames)

    # log video to run
    wbu.log_moviepy_clip("video", pil_frames_to_clip(video_frames), fps=10)

    # save video artifact
    video_artifact.log()
