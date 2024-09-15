import re
from pathlib import Path
from re import Pattern
import tempfile
from pytorch3d.renderer import FoVPerspectiveCameras
import torch
from tqdm import tqdm
from wandb import CommError, Artifact
from wandb.apis.public import Run, File
from PIL import Image
import shutil


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
        features: list[torch.Tensor],
        images: list[Image.Image],
    ) -> Artifact:

        # create temproary directory
        tempdir = tempfile.mkdtemp()

        # save cameras
        tempdir_path = Path(tempdir)
        torch.save(cameras, tempdir_path / 'cameras.pt')
        for i in range(len(cameras)):
            view_path = tempdir_path / f'view_{i}'
            view_path.mkdir(parents=True, exist_ok=True)
            torch.save(features[i], view_path / 'features.pt')
            images[i].save(view_path / 'image.png')

        artifact = Artifact(artifact_name, type=MVFeaturesArtifact.type)
        artifact.add_dir(tempdir_path)

        shutil.rmtree(tempdir)

        return artifact

    def __init__(self, artifact: Artifact):
        self.artifact = artifact
        self.path = Path(artifact.download())

    def get_cameras(self):
        return torch.load(self.path / 'cameras.pt')

    def get_features(self):
        features = []
        for i in range(len(self.get_cameras())):
            view_dir = self.path / f'view_{i}'
            features.append(torch.load(view_dir / 'features.pt'))
        return features


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

    def get_animation_path(self) -> Path:
        return self.path / 'animation.obj'
