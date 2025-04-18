import torch
import torchvision.transforms.functional as TF
from attr import dataclass

import wandb
import wandb_util.wandb_util as wbu
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.artifacts.texture_artifact import TextureArtifact
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.backprojection import (
    aggregate_views_uv_texture,
    project_visible_texels_to_camera,
)


@dataclass
class MakeTextureConfig:
    video_anim: str


@wbu.wandb_run("make_texture")
def make_rgb_texture(cfg: MakeTextureConfig, run_config: wbu.RunConfig):
    # disable gradients
    torch.set_grad_enabled(False)

    # get frames
    video = VideoArtifact.from_wandb_artifact_tag(cfg.video_anim, download=True)
    frames = video.read_frames()

    # read texturing views
    texgen_run = video.logged_by()

    texturing_anim_art = wbu.used_artifacts(texgen_run, type="animation")[0]
    texturing_anim_art = AnimationArtifact.from_wandb_artifact(texturing_anim_art)
    cams, meshes = texturing_anim_art.load_frames()
    verts_uvs, faces_uvs = texturing_anim_art.uv_data()

    # compute projections
    texture_res = 600
    projections = [
        project_visible_texels_to_camera(m, c, verts_uvs, faces_uvs, texture_res)
        for m, c in zip(meshes, cams)
    ]
    xys = [p.xys for p in projections]
    uvs = [p.uvs for p in projections]

    # create texture
    texturing_frames_pt = [TF.to_tensor(f) for f in frames]
    texturing_frames_pt = torch.stack(texturing_frames_pt).cuda()
    texture = aggregate_views_uv_texture(texturing_frames_pt, texture_res, xys, uvs)

    # log texture
    wandb.log({"texture": wandb.Image(texture.cpu().numpy())})

    texture_art = TextureArtifact.create_empty_artifact("texture")
    texture_art.write_texture(texture)
    texture_art.log_if_enabled()
