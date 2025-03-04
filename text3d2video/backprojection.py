from typing import Callable, Dict, List, Tuple

import torch
from einops import rearrange
from jaxtyping import Float
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    CamerasBase,
    MeshRasterizer,
    RasterizationSettings,
    TexturesUV,
    TexturesVertex,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes
from torch import Tensor

from text3d2video.rendering import TextureShader, make_mesh_rasterizer
from text3d2video.util import sample_feature_map_ndc, unique_with_indices

# Inverse Rendering


def project_visible_verts_to_camera(
    meshes: Meshes, cameras: CamerasBase, raster_res=600
):
    # rasterize mesh, to get visible verts
    raster_settings = RasterizationSettings(
        image_size=raster_res,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(meshes)

    pix_to_face = fragments.pix_to_face[0]
    mask = pix_to_face > 0
    visible_face_indices = pix_to_face[mask]
    visible_face_indices = visible_face_indices.unique()
    visible_faces = meshes.faces_packed()[visible_face_indices]

    visible_vert_indices = torch.unique(visible_faces)
    visible_verts = meshes.verts_packed()[visible_vert_indices]

    visible_verts_ndc = cameras.transform_points_ndc(visible_verts)
    visible_verts_xy = visible_verts_ndc[:, 0:2]
    visible_verts_xy[:, 0] *= -1

    vert_indices = visible_vert_indices

    return visible_verts_xy, vert_indices


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

    # get visible pixels mask
    mask = fragments.zbuf[:, :, :, 0] > 0

    # for each face, for each vert, uv coord
    faces_verts_uvs = verts_uvs[faces_uvs]

    # interpolate uv coordinates, to get a uv at each pixel
    pixel_uvs = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
    )
    pixel_uvs = pixel_uvs[:, :, :, 0, :]

    # 2D coordinate for each pixel, NDC space
    xs = torch.linspace(-1, 1, raster_res)
    ys = torch.linspace(1, -1, raster_res)
    ndc_xs, ndc_ys = torch.meshgrid(xs, ys, indexing="xy")
    ndc_coords = torch.stack([ndc_xs, ndc_ys], dim=-1).to(verts_uvs)

    # for each visible pixel get uv coord and xy coord
    uvs = pixel_uvs[mask]
    xys = ndc_coords[mask[0]]

    # convert continuous cartesian uv coords to pixel coords
    size = (texture_res, texture_res)
    uv_pix_coords = uvs.clone()
    uv_pix_coords[:, 0] = uv_pix_coords[:, 0] * size[0]
    uv_pix_coords[:, 1] = (1 - uv_pix_coords[:, 1]) * size[1]
    uv_pix_coords = uv_pix_coords.long()

    # remove duplicates, and get xy coords for each uv pixel
    texel_coords, coord_pix_indices = unique_with_indices(uv_pix_coords, dim=0)
    texel_xy_coords = xys[coord_pix_indices, :]

    return texel_xy_coords, texel_coords


# Aggregation


def aggregate_views_vert_texture(
    feature_maps: Float[Tensor, "n c h w"],
    n_verts: int,
    vertex_positions: List[Float[Tensor, "v 3"]],
    vertex_indices: List[Float[Tensor, "v"]],  # noqa: F821
    mode="nearest",
    aggregation_type="first",
):
    # initialize empty vertex features
    feature_dim = feature_maps.shape[1]
    vert_features = torch.zeros(n_verts, feature_dim).cuda()
    vert_features_cnt = torch.zeros(n_verts).cuda()

    for i, feature_map in enumerate(feature_maps):
        # get features for each vertex, for given view
        frame_vert_xys = vertex_positions[i]
        frame_vert_indices = vertex_indices[i]

        # project frame features to vertices
        frame_vert_features = sample_feature_map_ndc(
            feature_map,
            frame_vert_xys,
            mode=mode,
        ).to(vert_features)

        if aggregation_type == "mean":
            vert_features[frame_vert_indices] += frame_vert_features

        elif aggregation_type == "first":
            # update empty entries in vert_features
            mask = torch.all(vert_features[frame_vert_indices] == 0, dim=1)
            vert_features[frame_vert_indices[mask]] = frame_vert_features[mask]

        # count number of features per vertex
        frame_nonzero_features = torch.all(frame_vert_features != 0, dim=1)
        frame_nonzero_indices = frame_vert_indices[frame_nonzero_features]
        vert_features_cnt[frame_nonzero_indices] += 1

    if aggregation_type == "mean":
        vert_features /= torch.clamp(vert_features_cnt, min=1).unsqueeze(1)

    # average features
    return vert_features


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
    counts = torch.zeros(uv_resolution, uv_resolution)

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
        uv_map[uv_pix_coords[:, 1], uv_pix_coords[:, 0]] = colors
        counts[uv_pix_coords[:, 1], uv_pix_coords[:, 0]] += 1

    # average features

    # clamp so lowest count is 1
    counts_prime = torch.clamp(counts, min=1)
    uv_map /= counts_prime.unsqueeze(-1)

    return uv_map


# GR Aggregation, combine mean and inpainted texture


def gr_aggregate_views_uv_texture(
    feature_maps: Float[Tensor, "n c h w"],
    uv_resolution: int,
    texel_xys: List[Float[Tensor, "v 3"]],
    texel_uvs: List[Float[Tensor, "v"]],  # noqa: F821
    interpolation_mode="bilinear",
    mean_weight=0.5,
):
    mean_texture = aggregate_views_uv_texture_mean(
        feature_maps, uv_resolution, texel_xys, texel_uvs, interpolation_mode
    )
    inpainted_texture = aggregate_views_uv_texture(
        feature_maps, uv_resolution, texel_xys, texel_uvs, interpolation_mode
    )

    w_mean = mean_weight
    w_inpainted = 1 - w_mean
    texture = w_mean * mean_texture + w_inpainted * inpainted_texture

    return texture


def gr_aggregate_views_vert_texture(
    feature_maps: Float[Tensor, "n c h w"],
    n_verts: int,
    vertex_positions: List[Float[Tensor, "v 3"]],
    vertex_indices: List[Float[Tensor, "v"]],  # noqa: F821
    mean_weight=0.5,
):
    vert_ft_mean = aggregate_views_vert_texture(
        feature_maps,
        n_verts,
        vertex_positions,
        vertex_indices,
        mode="bilinear",
        aggregation_type="mean",
    )

    vert_ft_first = aggregate_views_vert_texture(
        feature_maps,
        n_verts,
        vertex_positions,
        vertex_indices,
        mode="bilinear",
        aggregation_type="first",
    )

    w_mean = mean_weight
    w_inpainted = 1 - w_mean
    vert_features = w_mean * vert_ft_mean + w_inpainted * vert_ft_first

    return vert_features


# Rendering


def render_vert_features(vert_features: Tensor, meshes: Meshes, fragments: Fragments):
    vert_features = vert_features.unsqueeze(0).expand(len(meshes), -1, -1)
    texture = TexturesVertex(vert_features)
    render_meshes = meshes.clone()
    render_meshes.textures = texture
    shader = TextureShader("cuda")
    render = shader(fragments, render_meshes)
    render = rearrange(render, "b h w c -> b c h w")

    return render


def rasterize_and_render_vt_features(
    vert_features: Tensor, meshes: Meshes, cams: CamerasBase, resolution: int
):
    raster_settings = RasterizationSettings(
        image_size=resolution, faces_per_pixel=1, bin_size=0
    )
    rasterizer = MeshRasterizer(cameras=cams, raster_settings=raster_settings)

    fragments = rasterizer(meshes, cameras=cams)
    render = render_vert_features(vert_features, meshes, fragments)

    return render


def rasterize_and_render_texture(
    texture: TexturesUV, meshes: Meshes, cams: CamerasBase
):
    render_meshes = meshes.clone()
    render_texture = texture.clone()
    render_texture = render_texture.extend(len(meshes))
    render_meshes.textures = render_texture

    rasterizer = make_mesh_rasterizer()
    fragments = rasterizer(render_meshes, cameras=cams)

    shader = TextureShader()
    render = shader(fragments, render_meshes)
    render = rearrange(render, "b h w c -> b c h w")

    return render


# all operate on dicts


def diffusion_dict_map(
    all_features: Dict[str, Tensor], f: Callable
) -> Dict[str, Tensor]:
    all_out_features = {}

    """
    Utility for applying a per-element function to a dictionary of features for each frame
    """

    # iterate over modules
    for module, module_features in all_features.items():
        stacked_out_features = []
        # iterate over batches
        for batch_features in module_features:
            # apply function to batch of features
            out_features = f(batch_features, module)
            stacked_out_features.append(out_features)
        stacked_out_features = torch.stack(stacked_out_features)
        all_out_features[module] = stacked_out_features
    return all_out_features


def aggregate_spatial_features_dict(
    spatial_diffusion_features: Dict[str, Float[Tensor, "n c h w"]],
    n_vertices: int,
    vertex_positions: List[Float[Tensor, "v 3"]],
    vertex_indices: List[Float[Tensor, "v"]],  # noqa: F821
) -> Dict[str, Float[Tensor, "b v c"]]:
    return diffusion_dict_map(
        spatial_diffusion_features,
        lambda feature_maps, _: gr_aggregate_views_vert_texture(
            feature_maps, n_vertices, vertex_positions, vertex_indices
        ),
    )


def rasterize_and_render_vert_features_dict(
    aggregated_features: Dict[str, Float[Tensor, "b v c"]],
    meshes: Meshes,
    cams: CamerasBase,
    resolutions: Dict[str, int] = None,
):
    return diffusion_dict_map(
        aggregated_features,
        lambda vt_ft, module: rasterize_and_render_vt_features(
            vt_ft, meshes, cams, resolution=resolutions[module]
        ),
    )
