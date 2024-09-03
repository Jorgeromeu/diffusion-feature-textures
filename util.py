import torch
from pytorch3d.renderer import CamerasBase
from pytorch3d.structures import Meshes
from torch import Tensor

def ndc_grid(
        resolution=100,
        corner_aligned=False
):
    """
    Return a 2xHxH tensor where each pixel has the NDC coordinates of the pixel
    :param resolution:
    :param corner_aligned
    :return:
    """

    u = 1 if corner_aligned else 1 - (1 / resolution)

    xs = torch.linspace(u, -u, resolution)
    ys = torch.linspace(u, -u, resolution)
    x, y = torch.meshgrid(xs, ys, indexing='xy')

    # stack to two-channel image
    xy = torch.stack([x, y])

    return xy

def reproject_features(
        cameras: CamerasBase,
        depth: Tensor,
        feature_map: Tensor
):

    H, W = depth.shape

    # 2D grid with ndc coordinate at each pixel
    xy_ndc = ndc_grid(H)

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
        xy_depth_points,
        world_coordinates=True,
        scaled_depth_input=False
    )

    return world_coords, point_features

def project_feature_map_to_vertices(
        mesh: Meshes,
        cam: CamerasBase,
        depth: Tensor,
        feature_map: Tensor,
        vertex_features: Tensor = None,
        distance_epsilon=0.1
):
    # extract resolution from depth map
    F = feature_map.shape[0]

    # if no initial features provided, initialize to zeros
    if vertex_features is None:
        vertex_features = torch.zeros(mesh.num_verts_per_mesh()[0], 3)

    world_coords, point_features = reproject_features(cam, depth, feature_map)

    # for each vertex, find the closest reprojected point
    # TODO replace with ball_query or KDTree
    distances = torch.cdist(
        world_coords,
        mesh.verts_list()[0],
        p=2
    )
    vertex_closest_point_distances, vertex_closest_point = torch.min(distances, dim=0)
    close_vertex_indices = vertex_closest_point_distances < distance_epsilon

    # for each vertex that has a projected point close to it, assign the nearest point feature
    vertex_features[close_vertex_indices] = point_features[vertex_closest_point[close_vertex_indices]]
    return vertex_features
