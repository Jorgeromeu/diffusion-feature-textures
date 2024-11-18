import shutil
from pathlib import Path
from typing import Tuple

import torch
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import CamerasBase
from pytorch3d.structures import Meshes
from torch import Tensor

from text3d2video.camera_placement import front_camera
from text3d2video.rendering import render_depth_map
from text3d2video.util import ordered_sample
from text3d2video.video_util import pil_frames_to_clip
from text3d2video.wandb_util import ArtifactWrapper


class AnimationArtifact(ArtifactWrapper):
    wandb_artifact_type = "animation"

    def write_frames(self, animation_path: str):
        # copy frames
        animation_dir = self._animation_path()
        animation_dir.mkdir()
        for frame in Path(animation_path).iterdir():
            number = frame.stem[-4:]
            frame_name = f"animation{number}.obj"
            shutil.copy(frame, animation_dir / frame_name)

    def write_unposed(self, unposed_path: str):
        shutil.copy(unposed_path, self.unposed_mesh_path())

    def write_cameras(self, cameras: CamerasBase):
        torch.save(cameras, self._cameras_path())

    # reading methods

    def _animation_path(self) -> Path:
        return self.folder / "animation"

    def _cameras_path(self) -> Path:
        return self.folder / "cameras.pt"

    def _has_cameras(self) -> bool:
        return self._cameras_path().exists()

    def _has_single_mesh(self) -> bool:
        return not self._animation_path().exists()

    # public methods

    def frame_nums(self, sample_n=None):
        if self._has_single_mesh():
            cameras = torch.load(self._cameras_path())
            frame_nums = list(range(len(cameras)))
        else:
            frame_paths = (self.folder / "animation").iterdir()
            frame_nums = [int(path.stem[-4:]) for path in frame_paths]
            frame_nums = sorted(frame_nums)

        if sample_n is not None:
            frame_nums = ordered_sample(frame_nums, sample_n)

        return frame_nums

    def unposed_mesh_path(self) -> Path:
        return self.folder / "static.obj"

    def frame_path(self, frame=1) -> Path:
        return self.folder / "animation" / f"animation{frame:04}.obj"

    def load_unposed_mesh(self, device: str = "cuda") -> Meshes:
        return load_objs_as_meshes([self.unposed_mesh_path()], device=device)

    def texture_data(self) -> Tuple[Tensor, Tensor]:
        _, faces, aux = load_obj(self.unposed_mesh_path())
        verts_uvs = aux.verts_uvs
        faces_uvs = faces.textures_idx
        return verts_uvs, faces_uvs

    def load_frame(self, frame: int, device: str = "cuda") -> Meshes:
        if self._has_single_mesh():
            return self.load_unposed_mesh(device=device)

        return load_objs_as_meshes([self.frame_path(frame)], device=device)

    def load_frames(self, frame_indices=None, device: str = "cuda") -> Meshes:
        if frame_indices is None:
            frame_indices = self.frame_nums()

        if self._has_single_mesh():
            return self.load_unposed_mesh(device=device).extend(len(frame_indices))

        return load_objs_as_meshes(
            [self.frame_path(frame) for frame in frame_indices], device=device
        )

    def camera(self, frame: int):
        if not self._has_cameras():
            return front_camera(1)

        return torch.load(self._cameras_path())[frame]

    def cameras(self, frame_indices=None):
        if frame_indices is None:
            frame_indices = self.frame_nums()

        if not self._has_cameras():
            return front_camera(len(frame_indices))

        return torch.load(self._cameras_path())[frame_indices]

    def render_depth_clip(self, n_frames=None, fps=10):
        frame_nums = self.frame_nums(n_frames)
        frames = self.load_frames(frame_nums).cuda()
        cams = self.cameras(frame_nums).cuda()
        depth_maps = render_depth_map(frames, cams)
        clip = pil_frames_to_clip(depth_maps, fps=fps)
        return clip
