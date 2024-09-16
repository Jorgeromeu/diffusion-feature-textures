import re
from pathlib import Path
from re import Pattern
import tempfile
from typing import Dict, List
from pytorch3d.renderer import FoVPerspectiveCameras
import torch
from tqdm import tqdm
from wandb import CommError, Artifact
from wandb.apis.public import Run, File
from PIL import Image
from pytorch3d.io import load_objs_as_meshes
import shutil

from text3d2video.file_util import OBJAnimation


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
        features: List[Dict[str, torch.Tensor]],
        images: List[Image.Image],
    ) -> Artifact:

        # create temproary directory
        tempdir = tempfile.mkdtemp()
        tempdir_path = Path(tempdir)

        # save cameras
        torch.save(cameras, tempdir_path / 'cameras.pt')

        # for each view save data
        for i in range(len(cameras)):
            view_path = tempdir_path / f'view_{i}'
            view_path.mkdir(parents=True, exist_ok=True)

            # save generated image
            images[i].save(view_path / 'image.png')

            for name, feature in features[i].items():
                torch.save(feature, view_path / f'feature-{name}.pt')

        artifact = Artifact(artifact_name, type=MVFeaturesArtifact.type)
        artifact.add_dir(tempdir_path)

        shutil.rmtree(tempdir)

        return artifact

    def __init__(self, artifact: Artifact):
        self.artifact = artifact
        self.path = Path(artifact.download())

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

    def get_feature(self, view_i: int, identifier: Dict[str, str]):
        view_dir = self.path / f'view_{view_i}'

        feature_name = 'feature-'
        feature_keys = [f'{key}:{value}' for key, value in identifier.items()]
        feature_name += ','.join(feature_keys) + '.pt'

        return torch.Tensor(torch.load(view_dir / feature_name))


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
