from typing import List

import torch
from jaxtyping import Float
from pytorch3d.renderer import (
    CamerasBase,
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from torch import Tensor

from text3d2video.util import sample_feature_map_ndc


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
