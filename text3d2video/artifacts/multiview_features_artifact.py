import math
from typing import Any, Dict, List

import torch
from PIL import Image
from pytorch3d.renderer import FoVPerspectiveCameras

import text3d2video.wandb_util as wbu
from text3d2video.artifacts.animation_artifact import AnimationArtifact, ArtifactWrapper
from text3d2video.disk_multidict import TensorDiskMultiDict
from text3d2video.multidict import MultiDict


class MVFeaturesArtifact(ArtifactWrapper):
    wandb_artifact_type = "multiview_features"

    def create_features_disk_dict(self) -> TensorDiskMultiDict:
        return TensorDiskMultiDict(self.get_features_path())

    def save_views(
        self,
        cameras: FoVPerspectiveCameras,
        images: List[Image.Image],
    ):
        # save cameras
        torch.save(cameras, self.folder / "cameras.pt")

        # for each view save image
        for i in range(len(cameras)):
            images[i].save(self.folder / f"view_{i}.png")

    def get_animation_from_lineage(self) -> AnimationArtifact:
        log_run = self.wandb_artifact.logged_by()
        anim_artifact = wbu.first_used_artifact_of_type(
            log_run, AnimationArtifact.wandb_artifact_type
        )
        return AnimationArtifact.from_wandb_artifact(anim_artifact)

    def view_indices(self) -> List[int]:
        return range(len(self.get_cameras()))

    def get_cameras(self):
        return torch.load(self.folder / "cameras.pt")

    def get_ims(self):
        ims = []
        for i in self.view_indices():
            ims.append(Image.open(self.folder / f"view_{i}.png"))
        return ims

    def get_features_path(self):
        return self.folder / "features"

    def get_feature_shape(self, layer: str):
        t = self.get_key_values("timestep")[0]
        view = self.get_key_values("view")[0]
        shape = self.get_feature(view, {"layer": layer, "timestep": t}).shape
        return shape

    def get_resolution(self, identifier: Dict[str, Any]):
        feature = self.get_feature(0, identifier)
        return int(math.sqrt(feature.shape[0]))

    def get_feature_dim(self, identifier: Dict[str, Any]):
        return self.get_feature(0, identifier).shape[0]

    def get_feature(self, view_i: int, identifier: Dict[str, Any]):
        full_identifier = {**identifier, "view": view_i}
        filename = MultiDict.dict_to_str(full_identifier) + ".pt"
        feature_paht = self.get_features_path() / filename
        return torch.Tensor(torch.load(feature_paht))

    def get_feature_by_id(self, identifier: Dict[str, Any]):
        filename = MultiDict.dict_to_str(identifier) + ".pt"
        feature_path = self.get_features_path() / filename
        return torch.Tensor(torch.load(feature_path))

    def feature_ids(self):
        identifiers = []
        for path in self.get_features_path().iterdir():
            stem = path.stem
            identifier = MultiDict.str_to_dict(stem)
            identifiers.append(identifier)
        return identifiers

    def get_key_values(self, key: str):
        values = set()
        for identifier in self.feature_ids():
            values.add(identifier[key])
        return list(values)
