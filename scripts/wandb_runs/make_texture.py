from dataclasses import dataclass

import torch
import torchvision.transforms.functional as TF

import wandb
import wandb_util.wandb_util as wbu
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.artifacts.texture_artifact import TextureArtifact
from text3d2video.backprojection import (
    aggregate_views_uv_texture_mean,
    compute_texel_projections,
)
from text3d2video.pipelines.pipeline_utils import (
    ModelConfig,
    load_pipeline,
)
from text3d2video.pipelines.texturing_pipeline import TexturingConfig, TexturingPipeline
from text3d2video.utilities.video_util import pil_frames_to_clip


@dataclass
class MakeTextureConfig:
    prompt: str
    animation_tag: str
    model: ModelConfig
    texgen: TexturingConfig
    seed: int = 0
    texture_out_art: str = "texture"


@wbu.wandb_run("make_texture")
def make_texture(cfg: MakeTextureConfig, run_config: wbu.RunConfig):
    # disable gradients
    torch.set_grad_enabled(False)

    # read animation
    anim = AnimationArtifact.from_wandb_artifact_tag(cfg.animation_tag, download=True)
    seq = anim.read_anim_seq()

    # load pipeline
    device = torch.device("cuda")
    pipe = load_pipeline(
        TexturingPipeline, cfg.model.sd_repo, cfg.model.controlnet_repo
    )

    # set seed
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    images = pipe(
        cfg.prompt,
        seq,
        conf=cfg.texgen,
        generator=generator,
    )

    # log video to run
    wbu.log_moviepy_clip("video", pil_frames_to_clip(images), fps=10)

    # project to texture
    texture_res = cfg.texgen.uv_res
    projections = compute_texel_projections(
        seq.meshes, seq.cams, seq.verts_uvs, seq.faces_uvs, texture_res
    )

    texturing_frames_pt = [TF.to_tensor(f) for f in images]
    texturing_frames_pt = torch.stack(texturing_frames_pt).cuda()
    texture = aggregate_views_uv_texture_mean(
        texturing_frames_pt, texture_res, projections
    )

    # log to run
    wandb.log({"texture": wandb.Image(texture.cpu().numpy())})

    # log as artifact
    texture_art = TextureArtifact.create_empty_artifact(cfg.texture_out_art)
    texture_art.write_texture(texture)
    texture_art.log()
