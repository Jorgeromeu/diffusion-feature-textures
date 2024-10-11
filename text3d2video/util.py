import math

import torch
import torch.nn.functional as F
from einops import rearrange
from pytorch3d.renderer import (CamerasBase, FoVPerspectiveCameras,
                                MeshRasterizer, RasterizationSettings,
                                look_at_view_transform)
from pytorch3d.structures import Meshes
from torch import Tensor


def ordered_sample(lst, N):
    """
    Sample N elements from a list in order.
    """

    step_size = len(lst) // (N - 1)
    # Get the sample by slicing the list
    sample = [lst[i * step_size] for i in range(N - 1)]
    sample.append(lst[-1])  # Add the last element
    return sample


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

    H, W = depth.shape

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


def project_feature_map_to_vertices(
    mesh: Meshes,
    cam: CamerasBase,
    depth: Tensor,
    feature_map: Tensor,
    vertex_features: Tensor = None,
    distance_epsilon=0.1,
):
    # extract resolution from depth map
    F = feature_map.shape[0]

    # if no initial features provided, initialize to zeros
    if vertex_features is None:
        vertex_features = torch.zeros(mesh.num_verts_per_mesh()[0], 3).to(feature_map)

    world_coords, point_features = reproject_features(cam, depth, feature_map)

    # for each vertex, find the closest reprojected point
    # TODO replace with ball_query or KDTree

    verts = mesh.verts_list()[0]

    distances = torch.cdist(world_coords.to(verts), verts, p=2)
    vertex_closest_point_distances, vertex_closest_point = torch.min(distances, dim=0)
    close_vertex_indices = vertex_closest_point_distances < distance_epsilon

    # for each vertex that has a projected point close to it, assign the nearest point feature
    vertex_features[close_vertex_indices] = point_features[
        vertex_closest_point[close_vertex_indices]
    ]
    return vertex_features


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


def sample_feature_map(feature_map: Tensor, coords: Tensor, mode="nearest"):
    batched_feature_map = rearrange(feature_map, "c h w -> 1 c h w").to(torch.float32)
    coords[:, 0] *= -1
    coords[:, 1] *= -1
    grid = rearrange(coords, "n d -> 1 1 n d")
    out = F.grid_sample(batched_feature_map, grid, align_corners=True, mode=mode)
    out_features = rearrange(out, "1 f 1 n -> n f")
    return out_features


def multiview_cameras(
    mesh: Meshes,
    num_views: int,
    add_angle_ele=0,
    add_angle_azi=0,
    scaling_factor=0.65,
    device="cpu",
) -> FoVPerspectiveCameras:
    """
    Generate cameras that envelope a mesh
    """

    # get bbox around mesh
    bbox = mesh.get_bounding_boxes()
    bbox_min = bbox.min(dim=-1).values[0]
    bbox_max = bbox.max(dim=-1).values[0]

    bb_diff = bbox_max - bbox_min
    bbox_center = (bbox_min + bbox_max) / 2.0
    distance = torch.sqrt((bb_diff * bb_diff).sum())
    distance *= scaling_factor

    steps = int(math.sqrt(num_views))
    end = 360 - 360 / steps
    elevation = (
        torch.linspace(start=0, end=end, steps=steps).repeat(steps) + add_angle_ele
    )
    azimuth = torch.linspace(start=0, end=end, steps=steps)
    azimuth = torch.repeat_interleave(azimuth, steps) + add_angle_azi
    bbox_center = bbox_center.unsqueeze(0)
    rotation, translation = look_at_view_transform(
        dist=distance, azim=azimuth, elev=elevation, device=device, at=bbox_center
    )

    cameras = FoVPerspectiveCameras(
        R=rotation, T=translation, device=device, znear=0.1, zfar=100
    )

    return cameras


def front_camera(n=1, device="cuda") -> FoVPerspectiveCameras:

    R, T = look_at_view_transform(dist=[2] * n, azim=[0] * n, elev=[0] * n)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60)
    return cameras
