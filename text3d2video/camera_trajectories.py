from pytorch3d.renderer import FoVOrthographicCameras, FoVPerspectiveCameras

from text3d2video.camera_placement import (
    fov_zoom,
    front_view_rotating,
    sideways_orthographic_cameras,
    turntable_cameras,
)


class NamedCameraTrajectory:
    """
    Base class for camera named, camera trajectories
    """

    name: str

    def cameras(self, n: int, device: str = "cuda") -> FoVPerspectiveCameras:
        pass


class RotationFull(NamedCameraTrajectory):
    name: str = "rotation"

    def cameras(self, n: int, device: str = "cuda") -> FoVPerspectiveCameras:
        return turntable_cameras(dist=1.2, n=n, device=device)


class RotationPartial(NamedCameraTrajectory):
    name: str = "rotation_partial"

    def cameras(self, n: int, device: str = "cuda") -> FoVPerspectiveCameras:
        return turntable_cameras(
            dist=1.2, n=n, start_angle=-30, stop_angle=30, device=device
        )


class Rotation90(NamedCameraTrajectory):
    name: str = "rotation_90"

    def cameras(self, n: int, device: str = "cuda") -> FoVPerspectiveCameras:
        return turntable_cameras(
            dist=1.2, n=n, start_angle=0, stop_angle=90, device=device
        )


class OrthographicPan(NamedCameraTrajectory):
    name: str = "orth_pan"

    def cameras(self, n: int, device: str = "cuda") -> FoVOrthographicCameras:
        return sideways_orthographic_cameras(n=n, x_0=0.6, x_1=-0.5, device=device)


class BarrelRoll(NamedCameraTrajectory):
    name: str = "barrel_roll"

    def cameras(self, n: int, device: str = "cuda") -> FoVPerspectiveCameras:
        return front_view_rotating(dist=1.1, n=n, device=device)


class FoVZoom(NamedCameraTrajectory):
    name: str = "fov_zoom"

    def cameras(self, n: int, device: str = "cuda") -> FoVPerspectiveCameras:
        return fov_zoom(n=n, z=1, fov_0=80, fov_1=20, device=device)


class BarrelRollPartial(NamedCameraTrajectory):
    name: str = "barrel_roll_partial"

    def cameras(self, n: int, device: str = "cuda") -> FoVPerspectiveCameras:
        span = 15
        return front_view_rotating(
            dist=1.1, start_angle=-span, stop_angle=span, n=n, device=device
        )
