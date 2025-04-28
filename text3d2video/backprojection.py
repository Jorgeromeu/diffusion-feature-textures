from typing import List, Tuple

import torch
from attr import dataclass
from jaxtyping import Float
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    CamerasBase,
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from torch import Tensor

from text3d2video.util import hwc_to_chw, sample_feature_map_ndc, unique_with_indices


@dataclass
class TexelProjection:
    xys: Tensor
    uvs: Tensor


def project_visible_texels_to_camera(
    mesh: Meshes,
    camera: CamerasBase,
    verts_uvs: Tensor,
    faces_uvs: Tensor,
    texture_res: int,
    raster_res=1000,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Project visible UV coordinates to camera pixel coordinates
    :param meshes: Meshes object
    :param cameras: CamerasBase object
    :param verts_uvs: (V, 2) UV coordinates
    :param faces_uvs: (F, 3) face indices for UV coordinates
    :param texture_res: UV_map resolution
    :param render_resolution: render resolution

    Return two corresponding sets of points in the camera and UV space

    :return xy_ndc_coords: (N, 2) pixel coordinates
    :return uv_pix_coords_unique: (N, 2) unique UV coordinates
    """

    # rasterize mesh at high resolution
    raster_settings = RasterizationSettings(
        image_size=raster_res,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    fragments = rasterizer(mesh, cameras=camera)

    # get UV coordinates for each pixel
    faces_verts_uvs = verts_uvs[faces_uvs]
    pixel_uvs = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
    )
    pixel_uvs = pixel_uvs[:, :, :, 0, :]

    # get visible pixels mask
    mask = fragments.zbuf[:, :, :, 0] > 0

    # 2D coordinate for each pixel, NDC space
    xs = torch.linspace(-1, 1, raster_res)
    ys = torch.linspace(1, -1, raster_res)
    ndc_xs, ndc_ys = torch.meshgrid(xs, ys, indexing="xy")
    ndc_coords = torch.stack([ndc_xs, ndc_ys], dim=-1).to(verts_uvs)

    # for each visible pixel get uv coord and xy coord
    uvs = pixel_uvs[mask]
    xys = ndc_coords[mask[0]]

    # convert continuous uv coords to texel coords
    size = (texture_res, texture_res)
    uv_pix_coords = uvs.clone()
    uv_pix_coords[:, 0] = uv_pix_coords[:, 0] * size[0]
    uv_pix_coords[:, 1] = (1 - uv_pix_coords[:, 1]) * size[1]
    uv_pix_coords = uv_pix_coords.long()

    # remove duplicates, and get xy coords for each uv pixel
    texel_coords, coord_pix_indices = unique_with_indices(uv_pix_coords, dim=0)
    texel_xy_coords = xys[coord_pix_indices, :]

    return TexelProjection(xys=texel_xy_coords, uvs=texel_coords)


# Aggregation


def aggregate_views_uv_texture(
    feature_maps: Float[Tensor, "n c h w"],
    uv_resolution: int,
    texel_xys: List[Float[Tensor, "v 3"]],
    texel_uvs: List[Float[Tensor, "v"]],  # noqa: F821
    interpolation_mode="bilinear",
):
    # initialize empty uv map
    feature_dim = feature_maps.shape[1]
    uv_map = torch.zeros(uv_resolution, uv_resolution, feature_dim).to(feature_maps)

    for i, feature_map in enumerate(feature_maps):
        xys = texel_xys[i]
        uvs = texel_uvs[i]

        # mask of unfilled texels
        mask = uv_map.sum(dim=-1) > 0
        mask = ~mask

        unfilled_indices = mask[uvs[:, 1], uvs[:, 0]]
        empty_uvs = uvs[unfilled_indices]
        empty_xys = xys[unfilled_indices]

        # sample features
        colors = sample_feature_map_ndc(
            feature_map,
            empty_xys,
            mode=interpolation_mode,
        ).to(uv_map)

        # update uv map
        uv_map[empty_uvs[:, 1], empty_uvs[:, 0]] = colors

    # average features
    return uv_map


def aggregate_views_uv_texture_mean(
    feature_maps: Float[Tensor, "n c h w"],
    uv_resolution: int,
    texel_xys: List[Float[Tensor, "v 3"]],
    texel_uvs: List[Float[Tensor, "v"]],  # noqa: F821
    interpolation_mode="bilinear",
):
    # initialize empty uv map
    feature_dim = feature_maps.shape[1]

    uv_map = torch.zeros(uv_resolution, uv_resolution, feature_dim).to(feature_maps)
    counts = torch.zeros(uv_resolution, uv_resolution).to(feature_maps.device)

    for i, feature_map in enumerate(feature_maps):
        xy_coords = texel_xys[i]
        uv_pix_coords = texel_uvs[i]

        # sample features
        colors = sample_feature_map_ndc(
            feature_map,
            xy_coords,
            mode=interpolation_mode,
        ).to(uv_map)

        # update uv map
        uv_map[uv_pix_coords[:, 1], uv_pix_coords[:, 0]] += colors
        counts[uv_pix_coords[:, 1], uv_pix_coords[:, 0]] += 1

    # clamp so lowest count is 1
    counts_prime = torch.clamp(counts, min=1)
    uv_map /= counts_prime.unsqueeze(-1)

    return uv_map


def update_uv_texture(
    uv_map: Tensor,
    feature_map: Tensor,
    texel_xys: Tensor,
    texel_uvs: Tensor,
    interpolation="bilinear",
    update_empty_only=True,
):
    # mask of unfilled texels

    if update_empty_only:
        mask = uv_map.sum(dim=-1) > 0
        mask = ~mask

        unfilled_indices = mask[texel_uvs[:, 1], texel_uvs[:, 0]]
        uvs = texel_uvs[unfilled_indices]
        xys = texel_xys[unfilled_indices]
    else:
        uvs = texel_uvs
        xys = texel_xys

    # sample features
    colors = sample_feature_map_ndc(
        feature_map,
        xys,
        mode=interpolation,
    ).to(uv_map)

    # update uv map
    uv_map[uvs[:, 1], uvs[:, 0]] = colors

    return uv_map


def project_view_to_texture_masked(
    uv_map: Tensor,
    feature_map: Float[Tensor, "c h w"],
    mask: Float[Tensor, "h w"],
    projection: TexelProjection,
):
    texels_in_mask = sample_feature_map_ndc(mask.unsqueeze(0).cuda(), projection.xys)
    uvs = projection.uvs[texels_in_mask[:, 0]]
    xys = projection.xys[texels_in_mask[:, 0]]

    # update texture
    colors = sample_feature_map_ndc(feature_map, xys).float()
    uv_map[uvs[:, 1], uvs[:, 0]] = colors
    return uv_map


def project_views_to_video_texture(
    feature_maps: Float[Tensor, "n c h w"],
    uv_resolution: int,
    projections: List[TexelProjection],
):
    xys = [p.xys for p in projections]
    uvs = [p.uvs for p in projections]

    texture_frames = []
    for i in range(len(feature_maps)):
        texture = torch.zeros(uv_resolution, uv_resolution, 3).to(feature_maps)
        view = feature_maps[i]
        update_uv_texture(texture, view, xys[i], uvs[i])
        texture_frames.append(texture)
    texture_frames = torch.stack(texture_frames, dim=0)
    texture_frames = hwc_to_chw(texture_frames)
    return texture_frames.cpu()
