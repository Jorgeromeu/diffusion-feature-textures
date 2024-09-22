import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from pytorch3d.renderer import FoVPerspectiveCameras
from wandb.apis.public import Run

from text3d2video.multidict import MultiDict
from wandb import Artifact


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
    type = "multiview_features"

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
        torch.save(cameras, tempdir_path / "cameras.pt")

        # for each view save image
        for i in range(len(cameras)):
            images[i].save(tempdir_path / "view_{i}.png")

        features_path = tempdir_path / "features"
        features_path.mkdir()
        features.serialize_multidict(features_path, extension="pt", save_fun=torch.save)

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
        return torch.load(self.path / "cameras.pt")

    def get_ims(self):
        ims = []
        for i in self.view_indices():
            view_dir = self.path / f"view_{i}"
            ims.append(Image.open(view_dir / "image.png"))
        return ims

    def get_feature(self, view_i: int, identifier: Dict[str, Any]):
        identifier.update({"view": view_i})
        filename = MultiDict.dict_to_str(identifier) + ".pt"
        feature_paht = self.path / "features" / filename
        return torch.Tensor(torch.load(feature_paht))
