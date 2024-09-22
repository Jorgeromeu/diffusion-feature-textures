from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from pytorch3d.renderer import FoVPerspectiveCameras

from text3d2video.animation_artifact import ArtifactWrapper
from text3d2video.multidict import MultiDict
from wandb import Artifact


class MVFeaturesArtifact(ArtifactWrapper):
    artifact_type = "multiview_features"

    @staticmethod
    def write_to_path(
        folder: Path,
        cameras: FoVPerspectiveCameras,
        features: MultiDict,
        images: List[Image.Image],
    ) -> Artifact:

        # save cameras
        torch.save(cameras, folder / "cameras.pt")

        # for each view save image
        for i in range(len(cameras)):
            images[i].save(folder / "view_{i}.png")

        features_path = folder / "features"
        features_path.mkdir()
        features.serialize_multidict(features_path, extension="pt", save_fun=torch.save)

    def view_indices(self) -> List[int]:
        return range(len(self.get_cameras()))

    def get_cameras(self):
        return torch.load(self.folder / "cameras.pt")

    def get_ims(self):
        ims = []
        for i in self.view_indices():
            view_dir = self.folder / f"view_{i}"
            ims.append(Image.open(view_dir / "image.png"))
        return ims

    def get_features_path(self):
        return self.folder / "features"

    def get_feature(self, view_i: int, identifier: Dict[str, Any]):
        identifier.update({"view": view_i})
        filename = MultiDict.dict_to_str(identifier) + ".pt"
        feature_paht = self.get_features_path() / filename
        return torch.Tensor(torch.load(feature_paht))