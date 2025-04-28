from collections import defaultdict
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

from text3d2video.backprojection import (
    project_visible_texels_to_camera,
    update_uv_texture,
)
from text3d2video.util import sample_feature_map_ndc


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


def normalize_depth_map(depth):
    """
    Convert from zbuf to depth map
    """

    max_depth = depth.max()
    indices = depth == -1
    depth = max_depth - depth
    depth[indices] = 0
    max_depth = depth.max()
    depth = depth / max_depth
    return depth


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
        depth_maps = normalize_depth_map(depth_maps)
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


def precompute_rasterization(
    cameras, meshes, vert_uvs, faces_uvs, render_resolutions, texture_resolutions
):
    projections = defaultdict(lambda: dict())
    fragments = defaultdict(lambda: dict())

    for frame_idx in range(len(cameras)):
        cam = cameras[frame_idx]
        mesh = meshes[frame_idx]

        for res_i in range(len(render_resolutions)):
            render_res = render_resolutions[res_i]
            texture_res = texture_resolutions[res_i]

            # project UVs to camera
            projection = project_visible_texels_to_camera(
                mesh,
                cam,
                vert_uvs,
                faces_uvs,
                raster_res=texture_res * 10,
                texture_res=texture_res,
            )

            # rasterize
            rasterizer = make_mesh_rasterizer(
                resolution=render_res,
                faces_per_pixel=1,
                blur_radius=0,
                bin_size=0,
            )
            frame_fragments = rasterizer(mesh, cameras=cam)

            fragments[frame_idx][res_i] = frame_fragments
            projections[frame_idx][res_i] = projection

    return projections, fragments


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


def shade_mesh(
    mesh,
    frags,
    uv_map,
    verts_uvs,
    faces_uvs,
):
    shader = TextureShader()
    texture = make_repeated_uv_texture(uv_map, faces_uvs, verts_uvs, N=1)
    render_mesh = mesh.clone()
    render_mesh.textures = texture
    render = shader(frags, render_mesh)
    return render[0]


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


def compute_newly_visible_masks(
    cams,
    meshes,
    projections,
    uv_res: int,
    image_res: int,
    verts_uvs,
    faces_uvs,
):
    """
    Given a sequence of cameras, compute the masks denoting for each render, the parts to update, according to what has already been seen, and image space coordinates
    """

    visible_texture = torch.ones(uv_res, uv_res, 1).cuda()
    visible_masks = []

    for i in range(len(cams)):
        proj = projections[i]
        mesh = meshes[i]
        cam = cams[i]

        # render visible mask
        mask_i = render_texture(
            mesh, cam, visible_texture, verts_uvs, faces_uvs, resolution=image_res
        )[0]
        visible_masks.append(mask_i[0].cpu())

        # update visible mask texture
        feature_map = torch.zeros(1, image_res, image_res).cuda()
        update_uv_texture(
            visible_texture,
            feature_map,
            proj.xys,
            proj.uvs,
            update_empty_only=False,
        )

    visible_masks = torch.stack(visible_masks)

    return visible_masks


def downsample_masks(masks: Tensor, size: Tuple[int], thresh=0.8):
    masks = torch.unsqueeze(masks, 1).float()
    masks_resized = TF.resize(masks, size, interpolation=TF.InterpolationMode.BILINEAR)
    masks_resized = masks_resized > thresh
    masks_resized = masks_resized.squeeze(1).cpu().float()
    return masks_resized
