import math

import torch
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes


def turntable_cameras(
    n: int, dist: float = 2, start_angle=0, stop_angle=360, device: str = "cuda"
) -> FoVPerspectiveCameras:
    azim = torch.linspace(start_angle, stop_angle, n)
    elev = [0] * n
    dists = [dist] * n
    R, T = look_at_view_transform(dists, elev, azim)
    return FoVPerspectiveCameras(device=device, R=R, T=T, fov=60)


def sideways_orthographic_cameras(x_0=1, x_1=-1, n: int = 10, device: str = "cuda"):
    line = torch.linspace(x_0, x_1, n)

    r = torch.eye(3)
    r[0, 0] = -1
    r[2, 2] = -1

    R = r.repeat(n, 1, 1)
    T = torch.zeros(n, 3)
    T[:, 2] = 2
    T[:, 0] = line

    cameras = FoVOrthographicCameras(device=device, R=R, T=T)

    return cameras


def z_movement_cameras(z_0=2, z_1=4, n: int = 10, device: str = "cuda"):
    line = torch.linspace(z_0, z_1, n)

    r = torch.eye(3)
    r[0, 0] = -1
    r[2, 2] = -1

    R = r.repeat(n, 1, 1)
    T = torch.zeros(n, 3)
    T[:, 2] = line

    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

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
