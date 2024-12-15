import torch
from torch import Tensor

# matrices from https://mkari.de/coord-converter/
BLENDER_WORLD_TO_PT3D_WORLD = Tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
BLENDER_CAM_TO_PT3D_CAM = Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])


def decompose_transform_srt(transform: Tensor, transposed=False):
    """
    Decompose a 4x4 homogeneous transform matrix representing a 3D transformation into its translation, scale and rotation components. Applied in the order: scale -> rotation -> translation.
    NOTE: only works for non-negative uniform scales
    """

    transform = transform.clone()

    if transposed:
        transform = transform.t()

    translation = transform[0:3, 3]
    mat_3x3 = transform[0:3, 0:3]
    scale_x = torch.norm(mat_3x3[0])
    scale_y = torch.norm(mat_3x3[1])
    scale_z = torch.norm(mat_3x3[2])
    scale = torch.stack([scale_x, scale_y, scale_z])
    rotation = mat_3x3 / scale
    return translation, scale, rotation


def assemble_transform_srt(translation: Tensor, scale: Tensor, rotation: Tensor):
    mat_3x3 = rotation * scale
    transform = torch.eye(4)
    transform[0:3, 3] = translation
    transform[0:3, 0:3] = mat_3x3
    return transform


def apply_transform_homogeneous(vertices: Tensor, transform: Tensor):
    verts_homog = torch.cat([vertices, torch.ones(vertices.shape[0], 1)], dim=1)
    vertices = verts_homog @ transform.t()
    return vertices[:, :3]
