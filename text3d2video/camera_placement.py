import math

import numpy as np
import torch
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    join_cameras_as_batch,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation

from text3d2video.coord_utils import assemble_transform_srt, decompose_transform_srt


def turntable_cameras(
    n: int,
    dist: float = 2,
    start_angle=0,
    stop_angle=360,
    fov=60,
    device: str = "cuda",
    endpoint=True,
    vertical=False,
) -> FoVPerspectiveCameras:
    azim = np.linspace(start_angle, stop_angle, n, endpoint=endpoint)
    elev = [0] * n

    if vertical:
        azim, elev = elev, azim

    dists = [dist] * n
    R, T = look_at_view_transform(dists, elev, azim)
    return FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov)


def sideways_orthographic_cameras(n: int = 10, x_0=1, x_1=-1, device: str = "cuda"):
    line = torch.linspace(x_0, x_1, n)

    r = torch.eye(3)
    r[0, 0] = -1
    r[2, 2] = -1

    R = r.repeat(n, 1, 1)
    T = torch.zeros(n, 3)
    T[:, 2] = 2
    T[:, 0] = line

    s = 1.6
    s_xyz = [(s, s, s)] * n

    cameras = FoVOrthographicCameras(device=device, R=R, T=T, scale_xyz=s_xyz)

    return cameras


def z_movement_cameras(n: int = 10, z_0=2, z_1=4, device: str = "cuda"):
    line = torch.linspace(z_0, z_1, n)

    r = torch.eye(3)
    r[0, 0] = -1
    r[2, 2] = -1

    R = r.repeat(n, 1, 1)
    T = torch.zeros(n, 3)
    T[:, 2] = line

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

    return cameras


def fov_zoom(n: int = 10, z=1, fov_0=80, fov_1=20, device: str = "cuda"):
    r = torch.eye(3)
    r[0, 0] = -1
    r[2, 2] = -1

    R = r.repeat(n, 1, 1)
    T = torch.zeros(n, 3)
    T[:, 2] = z

    fovs = torch.linspace(fov_0, fov_1, n)

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fovs)

    return cameras


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


def front_view_rotating(
    dist=1, n: int = 10, start_angle=0, stop_angle=360, device: str = "cuda"
) -> FoVPerspectiveCameras:
    t = torch.Tensor([0, 0, dist])
    angles = torch.linspace(start_angle, stop_angle, n)
    cams = []
    for angle in angles:
        r = Rotation.from_euler("yz", [180, angle], degrees=True)
        r = torch.Tensor(r.as_matrix())
        c2w = assemble_transform_srt(t, torch.ones(3), r)
        w2c = c2w.inverse()
        t_w2c, _, r_w2c = decompose_transform_srt(w2c)
        cam = FoVPerspectiveCameras(
            T=t_w2c.unsqueeze(0), R=r_w2c.unsqueeze(0), device=device
        )
        cams.append(cam)

    return join_cameras_as_batch(cams)
