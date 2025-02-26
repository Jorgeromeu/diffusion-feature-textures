import torch
from pytorch3d.structures import Meshes, join_meshes_as_batch
from torch import Tensor


def normalize_meshes(meshes: Meshes):
    """
    Normalize meshes by centering and scaling vertices to lie in unit cube
    """

    normalized_meshes = []
    for mesh in meshes:
        verts = mesh.verts_packed()
        verts_normalized = normalize_point_cloud(verts)
        normalized = Meshes(verts=[verts_normalized], faces=[mesh.faces_packed()])
        normalized_meshes.append(normalized)
    return join_meshes_as_batch(normalized_meshes)


def normalize_point_cloud(points: Tensor):
    """
    Normalize point cloud to be centered and lie in unit cube
    """

    min_vals = torch.min(points, dim=0).values
    max_vals = torch.max(points, dim=0).values
    bbox_center = (min_vals + max_vals) / 2
    max_extent = torch.max(max_vals - min_vals)

    translated = points - bbox_center
    scaled = translated / max_extent
    return scaled
