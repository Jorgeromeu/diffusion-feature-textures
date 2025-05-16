from typing import List, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange, repeat
from jaxtyping import Float
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    TexturesUV,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes
from torch import Tensor, nn

from text3d2video.util import sample_feature_map_ndc
from text3d2video.utilities.ipython_utils import display_ims


class TextureShader(nn.Module):
    """
    Simple shader, that returns textured colors of mesh, no shading
    """

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs):
        colors = meshes.sample_textures(fragments)
        mask = fragments.pix_to_face > 0
        output = torch.zeros_like(colors)
        output[mask] = colors[mask]
        output = output[:, :, :, 0, :]
        output = rearrange(output, "b h w c -> b c h w")
        return output


class UVShader(nn.Module):
    """
    Simple shader that returns the rendered UV coordiantes of mesh
    """

    def forward(self, fragments: Fragments, verts_uvs, faces_uvs):
        face_vert_uvs = verts_uvs[faces_uvs]
        pixel_uvs = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, face_vert_uvs
        )

        pixel_uvs = rearrange(pixel_uvs, "b h w 1 c -> b c h w")
        return pixel_uvs


def zbuf_to_depth_map(zbuf):
    """
    Convert from zbuf to depth map
    """

    max_depth = zbuf.max()
    indices = zbuf == -1
    zbuf = max_depth - zbuf
    zbuf[indices] = 0
    max_depth = zbuf.max()
    zbuf = zbuf / max_depth
    return zbuf


def make_mesh_rasterizer(
    cameras=None,
    resolution=512,
    faces_per_pixel=1,
    blur_radius=0,
    bin_size=0,
):
    raster_settings = RasterizationSettings(
        image_size=resolution,
        faces_per_pixel=faces_per_pixel,
        blur_radius=blur_radius,
        bin_size=bin_size,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    return rasterizer


def make_mesh_renderer(
    resolution=512,
    faces_per_pixel=1,
    blur_radius=0,
    bin_size=0,
    shader=None,
    cameras=None,
):
    rasterizer = make_mesh_rasterizer(
        resolution=resolution,
        faces_per_pixel=faces_per_pixel,
        blur_radius=blur_radius,
        bin_size=bin_size,
        cameras=cameras,
    )

    if shader is None:
        shader = TextureShader()

    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    return renderer


def render_depth_map(meshes, cameras, resolution=512, chunk_size=30):
    rasterizer = make_mesh_rasterizer(resolution=resolution)
    indices = torch.arange(0, len(meshes))

    all_depth_maps = []
    for chunk_indices in torch.split(indices, chunk_size):
        chunk_meshes = meshes[chunk_indices]
        chunk_cameras = cameras[chunk_indices]
        fragments = rasterizer(chunk_meshes, cameras=chunk_cameras)
        depth_maps = fragments.zbuf
        depth_maps = zbuf_to_depth_map(depth_maps)
        depth_maps = [TF.to_pil_image(depth_map[:, :, 0]) for depth_map in depth_maps]
        all_depth_maps.extend(depth_maps)

    return all_depth_maps


def render_rgb_uv_map(
    meshes, cameras, verts_uvs, faces_uvs, resolution=512, chunk_size=30
):
    rasterizer = make_mesh_rasterizer(resolution=resolution)
    shader = UVShader()

    all_uv_maps = []
    for i in range(0, len(meshes)):
        mesh = meshes[i]
        cam = cameras[i]
        fragments = rasterizer(mesh, cameras=cam)

        # render uv map
        pixel_uvs = shader(fragments, verts_uvs, faces_uvs)
        zero_channel = torch.zeros_like(pixel_uvs[:, 0:1, :, :])
        pixel_uvs_rgb = torch.cat([pixel_uvs, zero_channel], dim=1)

        frame_rgb = TF.to_pil_image(pixel_uvs_rgb[0])
        all_uv_maps.append(frame_rgb)

    return all_uv_maps


def make_repeated_uv_texture(
    uv_map: Float[Tensor, "h w c"],
    faces_uvs: Tensor,
    verts_uvs: Tensor,
    N=1,
    sampling_mode="bilinear",
):
    extended_uv_map = uv_map.to(torch.float32)  # pt3d requires float32 for textures
    extended_uv_map = extended_uv_map.unsqueeze(0).expand(N, -1, -1, -1)
    extended_faces_uvs = faces_uvs.unsqueeze(0).expand(N, -1, -1)
    extended_verts_uvs = verts_uvs.unsqueeze(0).expand(N, -1, -1)
    return TexturesUV(
        extended_uv_map,
        extended_faces_uvs,
        extended_verts_uvs,
        sampling_mode=sampling_mode,
    )


def shade_meshes(
    shader,
    texture: TexturesUV,
    meshes: Meshes,
    fragments: List[Fragments],
):
    renders = []
    for mesh, frags in zip(meshes, fragments):
        mesh.textures = texture
        render = shader(frags, mesh)[0]
        renders.append(render)

    renders = torch.stack(renders)

    return renders


def shade_texture(
    meshes,
    fragments,
    uv_map,
    verts_uvs,
    faces_uvs,
    sampling_mode="bilinear",
    return_pil=False,
):
    renders = []

    shader = TextureShader()
    meshes_copy = meshes.clone()
    texture = make_repeated_uv_texture(
        uv_map, faces_uvs, verts_uvs, sampling_mode=sampling_mode
    )
    for mesh, frags in zip(meshes_copy, fragments):
        mesh.textures = texture
        render = shader(frags, mesh)[0]
        renders.append(render)

    renders = torch.stack(renders)

    if return_pil:
        renders = [TF.to_pil_image(r.cpu()) for r in renders]

    return renders


def render_texture(
    meshes,
    cameras,
    uv_map,
    verts_uvs,
    faces_uvs,
    resolution=512,
    sampling_mode="bilinear",
    return_pil=False,
):
    n_frames = len(cameras)
    texture = make_repeated_uv_texture(
        uv_map, faces_uvs, verts_uvs, N=n_frames, sampling_mode=sampling_mode
    )
    renderer = make_mesh_renderer(cameras=cameras, resolution=resolution)
    meshes_copy = meshes.clone()
    meshes_copy.textures = texture
    renders = renderer(meshes_copy, cameras=cameras)

    if return_pil:
        renders = [TF.to_pil_image(r.cpu()) for r in renders]

    return renders


def compute_uv_jacobian_map(cam, mesh, verts_uvs, faces_uvs, res=512):
    rasterizer = make_mesh_rasterizer(resolution=res)
    frags = rasterizer(mesh, cameras=cam)

    face_vert_uvs = verts_uvs[faces_uvs]
    pixel_uvs = interpolate_face_attributes(
        frags.pix_to_face, frags.bary_coords, face_vert_uvs
    )

    pixel_uvs = rearrange(pixel_uvs, "b h w 1 c -> b c h w")

    bg_mask = rearrange(frags.pix_to_face == -1, "b h w 1 -> b 1 h w")
    bg_mask = repeat(bg_mask, "b 1 h w -> b c h w", c=2)
    bg_mask = -bg_mask.float() * 0

    pixel_uvs = pixel_uvs + bg_mask

    sobel_x = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        .view(1, 1, 3, 3)
        .to(pixel_uvs)
    )

    sobel_y = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        .view(1, 1, 3, 3)
        .to(pixel_uvs)
    )

    pixel_us = pixel_uvs[:, 0:1, :, :]
    pixel_vs = pixel_uvs[:, 1:2, :, :]

    du_dx = F.conv2d(pixel_us, sobel_x, padding=1)[0, 0].cpu()
    dv_dx = F.conv2d(pixel_vs, sobel_x, padding=1)[0, 0].cpu()
    du_dy = F.conv2d(pixel_us, sobel_y, padding=1)[0, 0].cpu()
    dv_dy = F.conv2d(pixel_vs, sobel_y, padding=1)[0, 0].cpu()

    abs_jacobian = torch.abs(du_dx * dv_dy - du_dy * dv_dx)

    return abs_jacobian


def compute_autoregressive_update_masks(
    cams,
    meshes,
    projections,
    quality_maps,
    uv_res: int,
    verts_uvs,
    faces_uvs,
    quality_factor=10,
):
    """
    Given a sequence of cameras, compute the masks denoting for each render, the parts to update, according to what has already been seen, and image space coordinates
    """

    quality_texture = torch.ones(uv_res, uv_res, 1).cuda() * 10000

    better_quality_masks = []

    for i in range(len(cams)):
        quality_map = quality_maps[i]
        proj = projections[i]
        mesh = meshes[i]
        cam = cams[i]

        # render quality texture
        rendered_quality = render_texture(
            mesh, cam, quality_texture, verts_uvs, faces_uvs
        )[0][0].cpu()

        # mask
        mask = quality_map < rendered_quality / quality_factor
        better_quality_masks.append(mask)

        # sample quality map for view i at prjoected texel locations
        qualities = sample_feature_map_ndc(quality_map.unsqueeze(0).cuda(), proj.xys)

        # get current-best qualities at projected texel locations
        current_qualities = quality_texture[proj.uvs[:, 1], proj.uvs[:, 0]]

        # mask indicating good quality
        better_quality = qualities > current_qualities
        update_mask = better_quality

        update_mask = ~update_mask
        update_mask = update_mask[:, 0]
        update_mask = update_mask

        # update qualities
        update_uvs = proj.uvs[update_mask]
        update_qualities = qualities[update_mask]
        quality_texture[update_uvs[:, 1], update_uvs[:, 0]] = update_qualities

    better_quality_masks = torch.stack(better_quality_masks)

    return better_quality_masks


def downsample_masks(masks: Tensor, size: Tuple[int], thresh=0.8):
    masks = torch.unsqueeze(masks, 1).float()
    masks_resized = TF.resize(masks, size, interpolation=TF.InterpolationMode.BILINEAR)
    masks_resized = masks_resized > thresh
    masks_resized = masks_resized.squeeze(1).cpu().float()
    return masks_resized


def display_frags(
    frags: Fragments,
    show_zbuf=True,
    show_pix_to_face=True,
    show_bary_coords=True,
    show_dists=True,
):
    ims = []
    titles = []

    if show_zbuf:
        ims.append(frags.zbuf[0, ..., 0].cpu())
        titles.append("zbuf")

    if show_pix_to_face:
        ims.append(frags.pix_to_face[0, ..., 0].cpu())
        titles.append("pix_to_face")

    if show_bary_coords:
        ims.append(frags.bary_coords[0, :, :, 0, :].cpu())
        titles.append("bary_coords")

    if show_dists:
        dists = frags.dists[0, ..., 0].cpu()
        ims.append(dists)
        titles.append("dists")

    display_ims(
        ims,
        titles=titles,
    )


def dilate_feature_map(feature_map, valid_mask, kernel_size=3, iterations=1):
    """
    Dilate features using max pooling on a valid mask and neighbor copying.
    Args:
        feature_map: (B, C, H, W) — features to dilate
        valid_mask:  (B, 1, H, W) — 1 where valid, 0 elsewhere
        kernel_size: convolution window for dilation
        iterations: how many times to apply dilation
    Returns:
        dilated_feature_map, dilated_mask
    """
    B, C, H, W = feature_map.shape

    # Copy input
    feat = feature_map.clone()
    mask = valid_mask.clone()

    for _ in range(iterations):
        # Get max mask in neighborhood: where new pixels will be added
        mask_dilated = F.max_pool2d(
            mask.float(), kernel_size, stride=1, padding=kernel_size // 2
        )
        new_mask = (mask_dilated > 0) & (mask == 0)  # new additions

        # For each channel, mask out invalid pixels and max-pool the rest
        feat_masked = feat * mask  # zero out invalid
        feat_sum = F.avg_pool2d(
            feat_masked, kernel_size, stride=1, padding=kernel_size // 2
        )
        norm = (
            F.avg_pool2d(mask.float(), kernel_size, stride=1, padding=kernel_size // 2)
            + 1e-6
        )
        feat_pooled = feat_sum / norm  # avoid divide by zero

        # Update only new pixels
        update = new_mask.expand(-1, C, -1, -1)
        feat[update] = feat_pooled[update]
        mask = mask | new_mask  # update mask

    return feat, mask


def dilate_frags(frags: Fragments, kernel_size=3, iterations=1):
    assert frags.pix_to_face.shape[-1] == 1, "only supports K=1"

    dilate_kwargs = {
        "kernel_size": kernel_size,
        "iterations": iterations,
    }

    pix_to_face = frags.pix_to_face[..., 0]

    valid = pix_to_face != -1
    valid_mask = valid.unsqueeze(1)

    pix_to_face_dil, mask_dil = dilate_feature_map(
        pix_to_face.unsqueeze(1).float(), valid_mask, **dilate_kwargs
    )
    pix_to_face_dil = pix_to_face_dil.squeeze(1).unsqueeze(-1).long()

    zbuf = frags.zbuf[..., 0]
    zbuf_dil, _ = dilate_feature_map(zbuf.unsqueeze(1), valid_mask, **dilate_kwargs)
    zbuf_dil = zbuf_dil.squeeze(1).unsqueeze(-1)

    bary_coords = rearrange(frags.bary_coords, "b h w 1 c -> b c h w")
    bary_coords_dil, _ = dilate_feature_map(bary_coords, valid_mask, **dilate_kwargs)
    bary_coords_dil = rearrange(bary_coords_dil, "b c h w -> b h w 1 c")

    dists = frags.dists[..., 0]
    dists_dil, _ = dilate_feature_map(dists.unsqueeze(1), valid_mask, **dilate_kwargs)
    dists_dil = rearrange(dists_dil, "b 1 h w -> b h w 1")

    frags_dil = Fragments(
        pix_to_face=pix_to_face_dil,
        zbuf=zbuf_dil,
        bary_coords=bary_coords_dil,
        dists=dists_dil,
    )
    return frags_dil
