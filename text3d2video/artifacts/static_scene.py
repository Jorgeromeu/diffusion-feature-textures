from pathlib import Path
from typing import Tuple

import torch
from pytorch3d.renderer import CamerasBase
from pytorch3d.structures import Meshes
from torch import Tensor

from text3d2video.wandb_util import ArtifactWrapper


class StaticSceneArtifact(ArtifactWrapper):
    wandb_artifact_type = "static_scene"

    # path methods
    def _mesh_path(self) -> Path:
        return self.folder / "meshes.pt"

    def _cam_path(self) -> Path:
        return self.folder / "cams.pt"

    def _verts_uvs_path(self) -> Path:
        return self.folder / "verts_uvs.pt"

    def _faces_uvs_path(self) -> Path:
        return self.folder / "faces_uvs.pt"

    # writing

    def write_scene(self, cam: CamerasBase, mesh: Meshes):
        assert len(cam) == len(mesh) == 1, "Number of cameras and meshes must be 1"
        torch.save(cam, self._cam_path())
        torch.save(mesh, self._mesh_path())

    def write_uv_data(self, verts_uvs: Tensor, faces_uvs: Tensor):
        torch.save(verts_uvs, self._verts_uvs_path())
        torch.save(faces_uvs, self._faces_uvs_path())

    # reading

    def uv_data(self) -> Tuple[Tensor, Tensor]:
        """
        Returns the UV data for the mesh.
        :return: Tuple of (verts_uvs, faces_uvs)
        """
        verts_uvs = torch.load(self._verts_uvs_path(), weights_only=False)
        faces_uvs = torch.load(self._faces_uvs_path(), weights_only=False)
        return verts_uvs, faces_uvs

    def load_scene(self, device: str = "cuda") -> Tuple[CamerasBase, Meshes]:
        cam = torch.load(self._cam_path(), weights_only=False)
        mesh = torch.load(self._mesh_path(), weights_only=False)
        return cam.to(device), mesh.to(device)
