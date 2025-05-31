from typing import List

import torch
from attr import dataclass
from jaxtyping import Float
from matplotlib import pyplot as plt
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    CamerasBase,
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.renderer.cameras import OrthographicCameras
from pytorch3d.structures import Meshes
from torch import Tensor

from text3d2video.rendering import (
    downsample_frags,
    make_mesh_rasterizer,
    render_texture,
)
from text3d2video.util import (
    hwc_to_chw,
    sample_feature_map_ndc,
    unique_with_indices,
)
from text3d2video.utilities.camera_placement import front_facing_extrinsics


@dataclass
class TexelProjection:
    """
    Texel Projection represents a mapping between UV coordinates and NDC coordinates.
    """

    xys: Tensor  # (N, 2) float pixel coordinates
    uvs: Tensor  # (N, 2) unique UV coordinates
    uv_resolution: int = 0  # resolution of the UV map


def compute_texel_projection_old(
    mesh: Meshes,
    camera: CamerasBase,
    verts_uvs: Tensor,
    faces_uvs: Tensor,
    texture_res: int,
    raster_res=1000,
) -> TexelProjection:
    """
    Project visible UV coordinates to camera pixel coordinates
    :param meshes: Meshes object
    :param cameras: CamerasBase object
    :param verts_uvs: (V, 2) UV coordinates
    :param faces_uvs: (F, 3) face indices for UV coordinates
    :param texture_res: UV_map resolution
    :param render_resolution: render resolution
    :return TexelProjection
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

    return TexelProjection(
        xys=texel_xy_coords, uvs=texel_coords, uv_resolution=texture_res
    )


def rasterize_uv_mesh(verts_uvs, faces_uvs, uv_res=512, blur_radius=0.0):
    verts_uvs_ndc = verts_uvs * 2 - 1
    verts_uvs_xyz = torch.cat(
        [verts_uvs_ndc, torch.zeros_like(verts_uvs[:, :1])], dim=-1
    )
    uv_mesh = Meshes(verts=[verts_uvs_xyz], faces=[faces_uvs])

    # rasterize uv mesh
    R, T = front_facing_extrinsics(zs=1)
    uv_camera = OrthographicCameras(R=R, T=T, device="cuda")
    uv_rasterizer = make_mesh_rasterizer(resolution=uv_res, blur_radius=blur_radius)

    return uv_rasterizer(uv_mesh, cameras=uv_camera)


def compute_texel_projection(
    mesh: Meshes,
    camera: CamerasBase,
    verts_uvs: Tensor,
    faces_uvs: Tensor,
    texture_res: int,
    raster_res=1000,
    visible_only=True,
) -> TexelProjection:
    """
    Project visible UV coordinates to camera pixel coordinates
    :param meshes: Meshes object
    :param cameras: CamerasBase object
    :param verts_uvs: (V, 2) UV coordinates
    :param faces_uvs: (F, 3) face indices for UV coordinates
    :param texture_res: UV_map resolution
    :param render_resolution: render resolution
    :return TexelProjection
    """

    factor = int(2000 / texture_res)
    if factor < 1:
        factor = 1
    factor = 1

    uv_frags = rasterize_uv_mesh(
        verts_uvs, faces_uvs, uv_res=texture_res * factor, blur_radius=1e-5
    )
    uv_frags = downsample_frags(uv_frags, factor=factor)

    pix_to_face = uv_frags.pix_to_face[0, ..., 0]
    bary_coords = uv_frags.bary_coords[0, :, :, 0, :]

    valid_mask = pix_to_face != -1

    faces = mesh.faces_list()[0]
    verts = mesh.verts_list()[0]

    # for each valid texel, its face index and bary coords
    texel_faces = pix_to_face[valid_mask]
    texel_bary = bary_coords[valid_mask]

    # get 3D coordinate of texel by interpolating the 3D triangles of the face
    tri_inds = faces[texel_faces.cpu()]
    v0, v1, v2 = verts[tri_inds[:, 0]], verts[tri_inds[:, 1]], verts[tri_inds[:, 2]]
    vert_coords_3d = (
        texel_bary[:, 0:1] * v0 + texel_bary[:, 1:2] * v1 + texel_bary[:, 2:3] * v2
    )

    # map verts to NDC space
    points_ndc = camera.transform_points_ndc(vert_coords_3d)
    points_ndc[:, 0] = -points_ndc[:, 0]  # flip y axis
    ndc_coords_xy = points_ndc[:, :2]

    # also map points to camera space to get their z-values (for depth test)
    points_cam = camera.get_world_to_view_transform().transform_points(vert_coords_3d)
    points_zs = points_cam[:, 2]

    # render zbuf for view
    depth_rasterizer = make_mesh_rasterizer(resolution=raster_res)
    view_frags = depth_rasterizer(mesh, cameras=camera)
    zbuf = view_frags.zbuf[0, ...]  # H W,1
    zbuf = hwc_to_chw(zbuf)  # 1 H W

    # get the z corresponding to texels from zbuf
    texel_closest_z = sample_feature_map_ndc(zbuf, ndc_coords_xy)[:, 0]

    eps = 0.01
    visible_mask = points_zs < texel_closest_z + eps

    # if no visibility check take all verts
    if not visible_only:
        visible_mask = torch.ones_like(visible_mask).bool()

    visible_xys = ndc_coords_xy[visible_mask]

    # finally get for each texel, its UV pixel coord
    u = torch.arange(0, texture_res)
    v = torch.arange(0, texture_res)
    U, V = torch.meshgrid(u, v, indexing="xy")
    uv = torch.stack([U, V], dim=-1).to(verts_uvs.device)
    uvs = uv[valid_mask]
    uvs = uvs[visible_mask]

    return TexelProjection(xys=visible_xys, uvs=uvs, uv_resolution=texture_res)


def compute_texel_projections(
    meshes: Meshes,
    cameras: CamerasBase,
    verts_uvs: Tensor,
    faces_uvs: Tensor,
    texture_res: int,
    raster_res=1000,
) -> List[TexelProjection]:
    """
    Compute texel projections for each camera
    :param meshes: Meshes object
    :param cameras: CamerasBase object
    :param verts_uvs: (V, 2) UV coordinates
    :param faces_uvs: (F, 3) face indices for UV coordinates
    :param texture_res: UV_map resolution
    :param render_resolution: render resolution
    :return List[TexelProjection]
    """
    projections = []
    for i in range(len(cameras)):
        projection = compute_texel_projection(
            meshes[i],
            cameras[i],
            verts_uvs,
            faces_uvs,
            texture_res,
            raster_res=raster_res,
        )
        projections.append(projection)
    return projections


def aggregate_views_uv_texture(
    feature_maps: Float[Tensor, "n c h w"],
    uv_resolution: int,
    projections: List[TexelProjection],
    interpolation_mode="bilinear",
):
    texel_xys = [projection.xys for projection in projections]
    texel_uvs = [projection.uvs for projection in projections]

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
    projections: List[TexelProjection],
    interpolation_mode="bilinear",
):
    texel_xys = [p.xys for p in projections]
    texel_uvs = [p.uvs for p in projections]

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
    projection: TexelProjection,
    interpolation="bilinear",
    update_empty_only=True,
):
    # mask of unfilled texels

    texel_xys = projection.xys
    texel_uvs = projection.uvs

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
    projections: List[TexelProjection],
):
    assert feature_maps.ndim == 4, "Feature maps must be 4D (T, C, H, W)"
    assert len(feature_maps) == len(
        projections
    ), "Number of feature maps and projections must match"

    uv_resolutions = [projection.uv_resolution for projection in projections]
    assert (
        len(set(uv_resolutions)) == 1
    ), "All projections must have the same resolution"
    uv_resolution = uv_resolutions[0]
    d = feature_maps.shape[1]

    texture_frames = []
    for i in range(len(feature_maps)):
        texture = torch.zeros(uv_resolution, uv_resolution, d).to(feature_maps)
        view = feature_maps[i]
        update_uv_texture(texture, view, projections[i])
        texture_frames.append(texture)
    texture_frames = torch.stack(texture_frames, dim=0)
    texture_frames = hwc_to_chw(texture_frames)
    return texture_frames.cpu()


def display_projection(
    projection: TexelProjection,
    cmap="turbo",
    s=0.5,
    alpha=0.2,
    uv_image=None,
    xy_image=None,
    scale=4,
):
    fig, axs = plt.subplots(1, 2, figsize=(2 * scale, 1 * scale))

    ax_xy = axs[0]
    ax_uv = axs[1]

    indices = torch.arange(0, projection.xys.shape[0])

    if xy_image is not None:
        ax_xy.imshow(xy_image, extent=(-1, 1, -1, 1))

    ax_xy.set_xlim(-1, 1)
    ax_xy.set_ylim(-1, 1)
    ax_xy.set_title("XY (NDC space)")

    ax_xy.scatter(
        projection.xys[:, 0].cpu(),
        projection.xys[:, 1].cpu(),
        s=s,
        c=indices.cpu(),
        cmap=cmap,
        alpha=alpha,
    )

    if uv_image is not None:
        ax_uv.imshow(uv_image)

    ax_uv.scatter(
        projection.uvs[:, 0].cpu(),
        projection.uvs[:, 1].cpu(),
        s=s,
        c=indices.cpu(),
        cmap=cmap,
        alpha=alpha,
    )
    ax_uv.set_title("UV")

    plt.tight_layout()
    pass


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
            proj,
            update_empty_only=False,
        )

    visible_masks = torch.stack(visible_masks)

    return visible_masks
