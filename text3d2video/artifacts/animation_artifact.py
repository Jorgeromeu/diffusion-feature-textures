import shutil
from pathlib import Path
from typing import Tuple

from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from torch import Tensor

from text3d2video.util import front_camera, ordered_sample
from text3d2video.wandb_util import ArtifactWrapper


class AnimationArtifact(ArtifactWrapper):

    wandb_artifact_type = "animation"

    def write_animation(self, animation_path: str, static_path: str):

        shutil.copy(static_path, self.unposed_mesh_path())

        # copy frames
        animation_dir = self.animation_path()
        animation_dir.mkdir()
        for frame in Path(animation_path).iterdir():
            number = frame.stem[-4:]
            frame_name = f"animation{number}.obj"
            shutil.copy(frame, animation_dir / frame_name)

    # reading methods

    def animation_path(self) -> Path:
        return self.folder / "animation"

    def unposed_mesh_path(self) -> Path:
        return self.folder / "static.obj"

    def frame_path(self, frame=1) -> Path:
        return self.folder / "animation" / f"animation{frame:04}.obj"

    def frame_nums(self, sample_n=None):

        frame_paths = (self.folder / "animation").iterdir()
        frame_nums = [int(path.stem[-4:]) for path in frame_paths]
        frame_nums = sorted(frame_nums)

        if sample_n is not None:
            frame_nums = ordered_sample(frame_nums, sample_n)

        return frame_nums

    def load_unposed_mesh(self, device: str = "cuda") -> Meshes:
        return load_objs_as_meshes([self.unposed_mesh_path()], device=device)

    def texture_data(self) -> Tuple[Tensor, Tensor]:
        frame_path = self.frame_path(self.frame_nums()[0])
        _, faces, aux = load_obj(frame_path)
        verts_uvs = aux.verts_uvs
        faces_uvs = faces.textures_idx
        return verts_uvs, faces_uvs

    def load_frame(self, frame: int, device: str = "cuda") -> Meshes:
        return load_objs_as_meshes([self.frame_path(frame)], device=device)

    def load_frames(self, frame_indices=None, device: str = "cuda") -> Meshes:

        if frame_indices is None:
            frame_indices = self.frame_nums()

        return load_objs_as_meshes(
            [self.frame_path(frame) for frame in frame_indices], device=device
        )

    # TODO include camera data in animation, for now just return front camera

    # pylint: disable=unused-argument
    def camera(self, frame: int):
        return front_camera(1)

    def cameras(self, frame_indices=None):

        if frame_indices is None:
            frame_indices = self.frame_nums()

        return front_camera(len(frame_indices))
