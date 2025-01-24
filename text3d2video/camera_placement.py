import math

import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
)
from pytorch3d.structures import Meshes
from scipy.spatial.transform import Rotation

from text3d2video.coord_utils import assemble_transform_srt, decompose_transform_srt


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


def broadcast_inputs(*args):
    args_tensors = []
    for arg in args:
        tensor = (
            torch.tensor(arg, dtype=torch.float32) if not torch.is_tensor(arg) else arg
        )
        args_tensors.append(tensor)

    if all([t.dim() == 0 for t in args_tensors]):
        args_tensors = [t.unsqueeze(0) for t in args_tensors]

    return torch.broadcast_tensors(*args_tensors)


def front_facing_extrinsics(degrees=0, xs=0, ys=0, zs=1.5):
    degrees, xs, ys, zs = broadcast_inputs(degrees, xs, ys, zs)

    ts = torch.stack([-xs, -ys, zs], dim=-1)

    rs = []
    for deg in degrees:
        r = Rotation.from_euler("yz", [180, deg], degrees=True)
        r = torch.Tensor(r.as_matrix())
        rs.append(r)
    rs = torch.stack(rs)

    ts_w2c = []
    rs_w2c = []
    for r, t in zip(rs, ts):
        c2w = assemble_transform_srt(t, torch.ones(3), r)
        w2c = c2w.inverse()
        t_w2c, _, r_w2c = decompose_transform_srt(w2c)
        ts_w2c.append(t_w2c)
        rs_w2c.append(r_w2c)

    return torch.stack(rs_w2c), torch.stack(ts_w2c)


def turntable_extrinsics(
    dists=1,
    angles=0,
    vertical=False,
) -> FoVPerspectiveCameras:
    dists, angles = broadcast_inputs(dists, angles)
    n = len(angles)

    azim = angles
    elev = [0] * n

    if vertical:
        azim, elev = elev, azim

    R, T = look_at_view_transform(dists, elev, azim)
    return R, T
