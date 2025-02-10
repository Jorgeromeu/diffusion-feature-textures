from typing import Callable, Dict, List

import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from jaxtyping import Float
from pytorch3d.renderer import (
    CamerasBase,
    MeshRasterizer,
    RasterizationSettings,
    TexturesVertex,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes
from torch import Tensor

from text3d2video.rendering import FeatureShader, make_rasterizer
from text3d2video.util import sample_feature_map_ndc
from text3d2video.utilities.ipython_utils import display_ims_grid


def project_visible_verts_to_cameras(meshes: Meshes, cameras: CamerasBase):
    # rasterize mesh, to get visible verts
    raster_settings = RasterizationSettings(
        image_size=600,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(meshes)

    vert_xys = []
    vert_indices = []
    packed_vert_cnt = 0

    for view in range(len(meshes)):
        pix_to_face = fragments.pix_to_face[view]
        mask = pix_to_face > 0
        visible_face_indices = pix_to_face[mask]
        visible_face_indices = visible_face_indices.unique()
        visible_faces = meshes.faces_packed()[visible_face_indices]

        visible_vert_indices = torch.unique(visible_faces)
        visible_verts = meshes.verts_packed()[visible_vert_indices]

        visible_verts_ndc = cameras[view].transform_points_ndc(visible_verts)
        visible_verts_xy = visible_verts_ndc[:, 0:2]
        visible_verts_xy[:, 0] *= -1

        vert_indices.append(visible_vert_indices - packed_vert_cnt)
        vert_xys.append(visible_verts_xy)

        packed_vert_cnt += meshes.num_verts_per_mesh()[view]

    return vert_xys, vert_indices


def aggregate_features_precomputed_vertex_positions(
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

    for frame, feature_map in enumerate(feature_maps):
        # get features for each vertex, for given view
        frame_vert_xys = vertex_positions[frame]
        frame_vert_indices = vertex_indices[frame]
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


def aggregate_spatial_features(
    feature_maps: Float[Tensor, "n c h w"],
    n_verts: int,
    vertex_positions: List[Float[Tensor, "v 3"]],
    vertex_indices: List[Float[Tensor, "v"]],  # noqa: F821
):
    vert_ft_mean = aggregate_features_precomputed_vertex_positions(
        feature_maps,
        n_verts,
        vertex_positions,
        vertex_indices,
        mode="bilinear",
        aggregation_type="mean",
    )

    vert_ft_first = aggregate_features_precomputed_vertex_positions(
        feature_maps,
        n_verts,
        vertex_positions,
        vertex_indices,
        mode="bilinear",
        aggregation_type="first",
    )

    w_mean = 0.5
    w_inpainted = 1 - w_mean
    vert_features = w_mean * vert_ft_mean + w_inpainted * vert_ft_first

    return vert_features


def render_vert_features(vert_features: Tensor, meshes: Meshes, fragments: Fragments):
    vert_features = vert_features.unsqueeze(0).expand(len(meshes), -1, -1)
    texture = TexturesVertex(vert_features)
    render_meshes = meshes.clone()
    render_meshes.textures = texture
    shader = FeatureShader("cuda")
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

    # render = TF.resize(
    #     render, resolution, interpolation=TF.InterpolationMode.NEAREST_EXACT
    # )

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
        lambda feature_maps, _: aggregate_spatial_features(
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


def display_spatial_features_dict(features, batch_idx=0, scale=2):
    ims_grid = []
    n_frames = None

    for layer in features.keys():
        ims = features[layer]
        ims = ims[batch_idx]
        n_frames = ims.shape[0]
        ims = [TF.to_pil_image(im) for im in ims]
        ims_grid.append(ims)

    display_ims_grid(
        ims_grid,
        col_titles=[int(i) for i in torch.arange(n_frames)],
        row_titles=layers,
        scale=scale,
        title=f"batch_idx: {batch_idx}",
    )
