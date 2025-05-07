from dataclasses import dataclass

import torch
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes

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
from wandb.apis.public import Run


@dataclass
class MakeTextureConfig:
    prompt: str
    animation_tag: str
    model: ModelConfig
    texgen: TexturingConfig
    seed: int = 0
    texture_out_art: str = "texture"


@dataclass
class MakeTextureData:
    prompt: str
    cams: CamerasBase
    meshes: Meshes
    verts_uvs: list
    faces_uvs: list
    texture: torch.Tensor

    @classmethod
    def from_run(cls, run: Run):
        # get prompt
        prompt = OmegaConf.create(run.config).prompt

        # get anim
        anim = wbu.used_artifacts(run, "animation")[0]
        anim = AnimationArtifact.from_wandb_artifact(anim)
        cams, meshes = anim.load_frames()
        verts_uvs, faces_uvs = anim.uv_data()

        # get texture
        tex_art = wbu.logged_artifacts(run, "rgb_texture")[0]
        tex_art = TextureArtifact.from_wandb_artifact(tex_art)
        texture = tex_art.read_texture()

        return cls(prompt, cams, meshes, verts_uvs, faces_uvs, texture)


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
    texture_res = 1000
    projections = compute_texel_projections(
        seq.meshes, seq.cams, seq.verts_uvs, seq.faces_uvs, texture_res, raster_res=2000
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
