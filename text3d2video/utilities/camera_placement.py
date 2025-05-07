import torch
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
)
from scipy.spatial.transform import Rotation

from text3d2video.utilities.coord_utils import (
    assemble_transform_srt,
    decompose_transform_srt,
)


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


def turntable_extrinsics(dists=2, angles=0, elevs=0) -> FoVPerspectiveCameras:
    dists, angles, elevs = broadcast_inputs(dists, angles, elevs)

    # target = torch.Tensor([0, 0.3, 0])
    # target = repeat(target, "d -> b d", b=dists.shape[0])

    R, T = look_at_view_transform(dists, elevs, angles)
    return R, T


def cam_view_prompt(angle, elev):
    if elev > 60:
        return "top"

    norm_angle = angle % 360

    if norm_angle >= 315 or norm_angle <= 45:
        return "front"
    elif norm_angle >= 135 and norm_angle <= 225:
        return "back"
    else:
        return "side"
