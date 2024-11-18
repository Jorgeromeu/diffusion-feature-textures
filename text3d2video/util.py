import itertools
from typing import Callable, List

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from pytorch3d.renderer import (
    CamerasBase,
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from torch import Tensor


def group_list_by(lst: List, key: Callable):
    sorted_list = sorted(lst, key=key)
    groupby_result = itertools.groupby(sorted_list, key)
    return [list(content) for _, content in groupby_result]


def ordered_sample(lst, n):
    """
    Sample n elements from a list in order.
    """
    if n <= 1:
        return [lst[0]] if n == 1 else []
    if n >= len(lst):
        return lst
    # Calculate the step size based on list length and number of samples
    step = (len(lst) - 1) / (n - 1)
    # Use the calculated step to select indices
    indices = [round(i * step) for i in range(n)]
    return [lst[i] for i in indices]


def pixel_coords_uv(
    res=100,
):
    xs = torch.linspace(0, 1, res)
    ys = torch.linspace(1, 0, res)
    x, y = torch.meshgrid(xs, ys, indexing="xy")

    return torch.stack([x, y])


def ndc_grid(resolution=100, corner_aligned=False):
    """
    Return a 2xHxH tensor where each pixel has the NDC coordinates of the pixel
    :param resolution:
    :param corner_aligned
    :return:
    """

    u = 1 if corner_aligned else 1 - (1 / resolution)

    xs = torch.linspace(u, -u, resolution)
    ys = torch.linspace(u, -u, resolution)
    x, y = torch.meshgrid(xs, ys, indexing="xy")

    # stack to two-channel image
    xy = torch.stack([x, y])

    return xy


def reproject_features(cameras: CamerasBase, depth: Tensor, feature_map: Tensor):
    H, _ = depth.shape

    # 2D grid with ndc coordinate at each pixel
    xy_ndc = ndc_grid(H).to(feature_map)

    # add depth channel
    xy_depth = torch.stack([xy_ndc[0, ...], xy_ndc[1, ...], depth], dim=-1)

    # mask selects all pixels that hit that correspond to a face
    mask = depth != -1

    # get NDC coords of points that hit the mesh
    # and their corresponding features from the feature map
    xy_depth_points = xy_depth[mask]
    point_features = feature_map[mask]

    # reproject NDC points back into 3D space
    world_coords = cameras[0].unproject_points(
        xy_depth_points, world_coordinates=True, scaled_depth_input=False
    )

    return world_coords, point_features


def project_vertices_to_features(
    mesh: Meshes, cam: CamerasBase, feature_map: Tensor, mode="nearest"
):
    feature_dim, _, _ = feature_map.shape

    # rasterize mesh
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(cameras=cam, raster_settings=raster_settings)
    fragments = rasterizer(mesh)

    # get visible faces
    visible_faces = rearrange(fragments.pix_to_face[0], "h w 1 -> (h w)").unique()

    # visible verts, are the vertices of the visible faces
    visible_vert_indices = mesh.faces_list()[0][visible_faces].flatten()
    verts = mesh.verts_list()[0]
    visible_verts = verts[visible_vert_indices]

    # TODO extract projected points from rasterization pass output, no need to do it again
    # project points to NDC
    visible_points_ndc = cam.transform_points_ndc(visible_verts).cpu()

    # extract features for each projected vertex
    visible_point_features = sample_feature_map(
        feature_map.cpu(), visible_points_ndc[:, 0:2].cpu(), mode
    ).to(verts)

    # construct vertex features tensor
    vert_features = torch.zeros(mesh.num_verts_per_mesh()[0], feature_dim).to(verts)
    vert_features[visible_vert_indices] = visible_point_features

    return vert_features


def project_vertices_to_cameras(meshes: Meshes, cameras: CamerasBase):
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
        frame_vert_features = sample_feature_map(
            feature_map.cpu(),
            frame_vert_xys.cpu(),
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


def sample_feature_map(feature_map: Tensor, coords: Tensor, mode="nearest"):
    """
    Sample the feature map at the given coordinates
    :param feature_map: (C, H, W) feature map
    :param coords: (N, 2) coordinates in the range [-1, 1] (NDC)
    :param mode: interpolation mode
    :return: (N, C) sampled features
    """
    coords = coords.clone()
    batched_feature_map = rearrange(feature_map, "c h w -> 1 c h w").to(torch.float32)
    grid = rearrange(coords, "n d -> 1 1 n d")
    out = F.grid_sample(batched_feature_map, grid, align_corners=True, mode=mode)
    out_features = rearrange(out, "1 f 1 n -> n f")
    return out_features.to(feature_map)


def blend_features(
    features_original: Tensor,
    features_rendered: Tensor,
    alpha: float,
    channel_dim=0,
):
    # compute mask, where features_rendered is not zero
    masks = torch.sum(features_rendered, dim=channel_dim, keepdim=True) != 0

    # blend features
    blended = alpha * features_rendered + (1 - alpha) * features_original

    original_background = features_original * ~masks
    blended_masked = blended * masks

    # return blended features, where features_rendered is not zero
    return original_background + blended_masked
