import torch
from pytorch3d.renderer import (
    CamerasBase,
    FoVOrthographicCameras,
    FoVPerspectiveCameras,
)

from text3d2video.utilities.camera_placement import (
    front_facing_extrinsics,
    turntable_extrinsics,
)


class NamedCameraTrajectory:
    """
    Base class for camera named, camera trajectories
    """

    name: str

    def cameras(self, n: int, device: str = "cuda") -> CamerasBase:
        pass


ORTH_S = 1.8
ROTATION_DIST = 1.2
ORTH_LO = -0.3
ORTH_HI = 0.3


class RotationFull(NamedCameraTrajectory):
    name: str = "rotation"

    def cameras(self, n: int, device: str = "cuda") -> CamerasBase:
        R, T = turntable_extrinsics(
            dists=ROTATION_DIST, angles=torch.linspace(0, 360, n)
        )
        return FoVPerspectiveCameras(R=R, T=T, device=device, fov=60)


class RotationPartial(NamedCameraTrajectory):
    name: str = "rotation_partial"

    def cameras(self, n: int, device: str = "cuda") -> CamerasBase:
        angles = torch.linspace(-30, 30, n)
        R, T = turntable_extrinsics(dists=ROTATION_DIST, angles=angles)
        return FoVPerspectiveCameras(R=R, T=T, device=device, fov=60)


class OrthographicPanHorizontal(NamedCameraTrajectory):
    name: str = "orth_pan"

    def cameras(self, n: int, device: str = "cuda") -> CamerasBase:
        s = ORTH_S
        xs = torch.linspace(ORTH_LO, ORTH_HI, n)
        R, T = front_facing_extrinsics(xs=xs)
        return FoVOrthographicCameras(R=R, T=T, device=device, scale_xyz=[(s, s, s)])


class BarrelRoll(NamedCameraTrajectory):
    name: str = "barrel_roll"

    def cameras(self, n: int, device: str = "cuda") -> CamerasBase:
        s = ORTH_S
        angles = torch.linspace(0, 360, n)
        R, T = front_facing_extrinsics(degrees=angles)
        return FoVOrthographicCameras(R=R, T=T, device=device, scale_xyz=[(s, s, s)])


class BarrelRollPartial(NamedCameraTrajectory):
    name: str = "barrel_roll_partial"

    def cameras(self, n: int, device: str = "cuda") -> CamerasBase:
        s = ORTH_S
        angles = torch.linspace(-30, 30, n)
        R, T = front_facing_extrinsics(degrees=angles)
        return FoVOrthographicCameras(R=R, T=T, device=device, scale_xyz=[(s, s, s)])


class BarrelRollPartialUpsideDown(NamedCameraTrajectory):
    name: str = "barrel_roll_partial_upside_down"

    def cameras(self, n: int, device: str = "cuda") -> CamerasBase:
        s = ORTH_S
        diff = 30
        angles = torch.linspace(180 - diff, 180 + diff, n)
        R, T = front_facing_extrinsics(degrees=angles)
        return FoVOrthographicCameras(R=R, T=T, device=device, scale_xyz=[(s, s, s)])
