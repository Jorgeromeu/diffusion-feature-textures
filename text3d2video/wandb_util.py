import re
from pathlib import Path
from re import Pattern
import tempfile
from typing import Any, Dict, List
from pytorch3d.renderer import FoVPerspectiveCameras
import torch
from tqdm import tqdm
from wandb import CommError, Artifact
from wandb.apis.public import Run, File
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
import shutil

from text3d2video.file_util import OBJAnimation
from text3d2video.multidict import MultiDict


def first_logged_artifact_of_type(run: Run, artifact_type: str) -> Artifact:
    for artifact in run.logged_artifacts():
        if artifact.type == artifact_type:
            return artifact
    return None


def first_used_artifact_of_type(run: Run, artifact_type: str) -> Artifact:
    for artifact in run.used_artifacts():
        if artifact.type == artifact_type:
            return artifact
    return None


class MVFeaturesArtifact:
    type = 'multiview_features'

    @staticmethod
    def create(
        artifact_name: str,
        cameras: FoVPerspectiveCameras,
        features: MultiDict,
        images: List[Image.Image],
    ) -> Artifact:

        # create temproary directory
        tempdir = tempfile.mkdtemp()
        tempdir_path = Path(tempdir)

        # save cameras
        torch.save(cameras, tempdir_path / 'cameras.pt')

        # for each view save image
        for i in range(len(cameras)):
            images[i].save(tempdir_path / 'view_{i}.png')

        features_path = tempdir_path / 'features'
        features_path.mkdir()
        features.serialize_multidict(
            features_path,
            extension='pt',
            save_fun=torch.save
        )

        artifact = Artifact(artifact_name, type=MVFeaturesArtifact.type)
        artifact.add_dir(tempdir_path)

        shutil.rmtree(tempdir)

        return artifact

    def __init__(self, artifact: Artifact):
        self.artifact = artifact
        self.path = Path(artifact.download())

    def _ident_dict_to_str(identifier: Dict[str, Any]) -> str:
        items = sorted(identifier.items())
        ident_str = [f'{k}:{v}' for k, v in items]
        ident_str = ','.join(ident_str)
        return ident_str

    def view_indices(self) -> List[int]:
        return range(len(self.get_cameras()))

    def get_cameras(self):
        return torch.load(self.path / 'cameras.pt')

    def get_ims(self):
        ims = []
        for i in self.view_indices():
            view_dir = self.path / f'view_{i}'
            ims.append(Image.open(view_dir / 'image.png'))
        return ims

    def get_feature(self, view_i: int, identifier: Dict[str, Any]):
        identifier.update({'view': view_i})
        filename = MultiDict._dict_to_str(identifier) + '.pt'
        feature_paht = self.path / 'features' / filename
        return torch.Tensor(torch.load(feature_paht))


class AnimationArtifact:

    type = 'animation'

    @staticmethod
    def create(artifact_name: str, animation_path: str, static_path: str) -> Artifact:
        artifact = Artifact(artifact_name, type=AnimationArtifact.type)
        artifact.add_dir(animation_path, name='animation')
        artifact.add_file(static_path, name='static.obj')
        return artifact

    def __init__(self, artifact: Artifact):
        self.artifact = artifact
        self.path = Path(artifact.download())

    def get_mesh_path(self) -> Path:
        return self.path / 'static.obj'

    def get_mesh(self, device='cuda:0'):
        return load_objs_as_meshes([self.get_mesh_path()], device=device)

    def get_animation(self) -> OBJAnimation:
        return OBJAnimation(self.path / 'animation')

    def get_animation_path(self) -> Path:
        return self.path / 'animation.obj'
