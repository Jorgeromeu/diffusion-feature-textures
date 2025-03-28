from pathlib import Path
from typing import Tuple

import torch
from attr import dataclass
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.renderer import CamerasBase
from pytorch3d.structures import Meshes
from torch import Tensor

from text3d2video.util import ordered_sample
from wandb_util.wandb_util import ArtifactWrapper


@dataclass
class AnimationConfig:
    n_frames: int
    artifact_tag: str


class MeshArtifact(ArtifactWrapper):
    wandb_artifact_type = "mesh"

    # path methods
    def _meshes_path(self) -> Path:
        return self.folder / "meshes.pt"

    def _verts_uvs_path(self) -> Path:
        return self.folder / "verts_uvs.pt"

    def _faces_uvs_path(self) -> Path:
        return self.folder / "faces_uvs.pt"

    def write_from_obj(self, obj_path: str):
        verts, faces, aux = load_obj(obj_path)
        verts_uvs = aux.verts_uvs
        faces_uvs = faces.textures_idx

        mesh = load_objs_as_meshes([obj_path])
        self.write_mesh(mesh)
        self.write_uv_data(verts_uvs, faces_uvs)

    # writing

    def write_mesh(self, meshes: Meshes):
        torch.save(meshes, self._meshes_path())

    def write_uv_data(self, verts_uvs: Tensor, faces_uvs: Tensor):
        torch.save(verts_uvs, self._verts_uvs_path())
        torch.save(faces_uvs, self._faces_uvs_path())

    # reading

    def mesh(self, device="cuda") -> Meshes:
        return torch.load(self._meshes_path()).to(device)

    def uv_data(self, device="cuda") -> Tuple[Tensor, Tensor]:
        verts_uvs = torch.load(self._verts_uvs_path(), weights_only=False)
        faces_uvs = torch.load(self._faces_uvs_path(), weights_only=False)
        return verts_uvs.to(device), faces_uvs.to(device)
