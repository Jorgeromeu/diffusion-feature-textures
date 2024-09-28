import os

import numpy as np
import rerun as rr
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d

PT3D_ViewCoords = rr.ViewCoordinates.LUF


class TimeSequence:
    """
    Utility class for time sequences in rerun
    """

    def __init__(self, name: str) -> None:
        self.cur_step = 0
        self.sequence_name = name
        rr.set_time_sequence(name, 0)

    def step(self):
        self.cur_step += 1
        rr.set_time_sequence(self.sequence_name, self.cur_step)


def set_logging_state(state: bool):
    value = "on" if state else "off"
    os.environ["RERUN"] = value


def feature_map(feature_map: np.array):
    """
    Log a high-dimensional feature map as a tensor
    :param feature_map: D x H x W tensor
    """
    dim_names = ["feature", "height", "width"]
    return rr.Tensor(feature_map, dim_names=dim_names)


def pt3d_setup():
    rr.log("/", PT3D_ViewCoords, static=True)


def pt3d_mesh(meshes: Meshes, batch_idx=0, vertex_colors=None):
    # extract verts and faces from idx-th mesh in batch
    verts = meshes.verts_list()[batch_idx].cpu()
    faces = meshes.faces_list()[batch_idx].cpu()
    vertex_normals = meshes.verts_normals_list()[batch_idx].cpu()

    # # TODO figure out hw to render textures...
    # tex = meshes.textures.maps_padded()[batch_idx].cpu().numpy()
    # uv = meshes.textures.verts_uvs_padded()[0].cpu().numpy()

    return rr.Mesh3D(
        vertex_positions=verts,
        triangle_indices=faces,
        vertex_normals=vertex_normals,
        vertex_colors=vertex_colors,
    )


def log_pt3d_FovCamrea(
    label: str, cameras: FoVPerspectiveCameras, batch_idx=0, res=100
):
    rr.log(label, pt3d_FovCamera(cameras, batch_idx, res))
    cam_trans = cameras.get_world_to_view_transform().inverse()
    rr.log(label, pt3d_transform(cam_trans, batch_idx))


def pt3d_FovCamera(cameras: FoVPerspectiveCameras, batch_idx=0, res=100):
    # TODO figure out how to get size from raster settings
    fov = cameras[batch_idx].fov.item()
    focal_length = int(res / (2 * np.tan(fov * np.pi / 360)))

    return rr.Pinhole(
        height=res,
        width=res,
        focal_length=focal_length,
        camera_xyz=PT3D_ViewCoords,
    )


def pt3d_transform(transforms: Transform3d, batch_idx=0):
    matrix = transforms.get_matrix()
    translation = matrix[batch_idx, 3, 0:3].cpu()
    rotation = matrix[batch_idx, 0:3, 0:3].cpu().inverse()
    return rr.Transform3D(translation=translation, mat3x3=rotation)
