from pathlib import Path
from typing import Tuple

import torch
from attr import dataclass
from pytorch3d.renderer import CamerasBase
from pytorch3d.structures import Meshes
from torch import Tensor

from text3d2video.rendering import render_depth_map, render_rgb_uv_map
from text3d2video.util import ordered_sample
from wandb_util.wandb_util import ArtifactWrapper


@dataclass
class AnimationConfig:
    n_frames: int
    artifact_tag: str


@dataclass
class AnimSequence:
    cams: CamerasBase
    meshes: Meshes
    verts_uvs: Tensor
    faces_uvs: Tensor

    def __len__(self):
        return len(self.cams)

    def render_depth_maps(self):
        """
        Returns a list of depth maps for each frame in the animation sequence.
        """
        return render_depth_map(self.meshes, self.cams)

    def render_rgb_uv_maps(self):
        return render_rgb_uv_map(self.meshes, self.cams, self.verts_uvs, self.faces_uvs)


class AnimationArtifact(ArtifactWrapper):
    wandb_artifact_type = "animation"

    # path methods
    def _meshes_path(self) -> Path:
        return self.folder / "meshes.pt"

    def _cams_path(self) -> Path:
        return self.folder / "cams.pt"

    def _verts_uvs_path(self) -> Path:
        return self.folder / "verts_uvs.pt"

    def _faces_uvs_path(self) -> Path:
        return self.folder / "faces_uvs.pt"

    # writing

    def write_frames(self, cams: CamerasBase, meshes: Meshes):
        assert len(cams) == len(meshes), "Number of cameras and meshes must match"
        torch.save(cams, self._cams_path())
        torch.save(meshes, self._meshes_path())

    def write_uv_data(self, verts_uvs: Tensor, faces_uvs: Tensor):
        torch.save(verts_uvs, self._verts_uvs_path())
        torch.save(faces_uvs, self._faces_uvs_path())

    def write_anim_seq(self, seq: AnimSequence):
        """
        Writes the animation sequence to the artifact.
        :param seq: AnimSequence object
        """
        self.write_frames(seq.cams, seq.meshes)
        self.write_uv_data(seq.verts_uvs, seq.faces_uvs)

    # reading

    def frame_indices(self, sample_n=None):
        """
        Returns indices of frames to render.
        if sample_n is not None, sample N evenly spaced frames
        """
        meshes = torch.load(self._meshes_path())
        indices = list(range(len(meshes)))
        if sample_n is not None:
            indices = ordered_sample(indices, sample_n)
        return indices

    def uv_data(self, device="cuda") -> Tuple[Tensor, Tensor]:
        """
        Returns the UV data for the mesh.
        :return: Tuple of (verts_uvs, faces_uvs)
        """
        verts_uvs = torch.load(self._verts_uvs_path(), weights_only=False)
        faces_uvs = torch.load(self._faces_uvs_path(), weights_only=False)
        return verts_uvs.to(device), faces_uvs.to(device)

    def load_frames(
        self, indices=None, device: str = "cuda"
    ) -> Tuple[CamerasBase, Meshes]:
        cams = torch.load(self._cams_path())
        meshes = torch.load(self._meshes_path())

        if indices is not None:
            cams = cams[indices]
            meshes = meshes[indices]
        return cams.to(device), meshes.to(device)

    def read_anim_seq(self, indices=None) -> AnimSequence:
        verts_uvs, faces_uvs = self.uv_data()
        cams, meshes = self.load_frames(indices)
        return AnimSequence(
            cams=cams, meshes=meshes, verts_uvs=verts_uvs, faces_uvs=faces_uvs
        )
