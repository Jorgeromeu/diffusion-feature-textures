from typing import List

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from pytorch3d.renderer import CamerasBase
from pytorch3d.structures import Meshes
from rerun import Tensor

from text3d2video.backprojection import (
    compute_texel_projections,
    project_views_to_video_texture,
)


def adjacent_frame_uv_mses(video_texture: Tensor):
    mses = []
    for i in range(len(video_texture) - 1):
        frame = video_texture[i]
        next_frame = video_texture[i + 1]

        # get visibility masks
        mask_0 = torch.sum(frame, dim=0) > 0
        mask_1 = torch.sum(next_frame, dim=0) > 0
        mask_both = mask_0 & mask_1

        pixels_frame = frame[:, mask_both]
        pixels_next_frame = next_frame[:, mask_both]
        mse = F.mse_loss(pixels_frame, pixels_next_frame)
        mses.append(mse.item())

    mses = torch.tensor(mses)
    return mses


def mean_uv_mse(
    frames: List[Image.Image],
    cameras: CamerasBase,
    meshes: Meshes,
    verts_uvs,
    faces_uvs,
    uv_res=600,
):
    frames_pt = [TF.to_tensor(f) for f in frames]
    frames_pt = torch.stack(frames_pt).cuda()

    # compute projections
    projections = compute_texel_projections(
        meshes, cameras, verts_uvs, faces_uvs, uv_res
    )

    # get video texture
    video_texture = project_views_to_video_texture(frames_pt, projections)

    # get average mse
    mses = adjacent_frame_uv_mses(video_texture)
    return torch.mean(mses).item()
