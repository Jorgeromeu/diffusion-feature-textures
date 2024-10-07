from pathlib import Path
from typing import Any, Dict

from text3d2video.disk_multidict import TensorDiskMultiDict
import text3d2video.wandb_util as wu
from text3d2video.artifacts.multiview_features_artifact import MVFeaturesArtifact
from text3d2video.wandb_util import ArtifactWrapper


class VertAttributesArtifact(ArtifactWrapper):

    wandb_artifact_type = "vertex_atributes"

    def get_features_disk_dict(self) -> TensorDiskMultiDict:
        return TensorDiskMultiDict(self.get_features_path())

    def get_features_path(self):
        return self.folder / "features"

    def get_mv_features_from_lineage(self) -> MVFeaturesArtifact:
        log_run = self.wandb_artifact.logged_by()
        mv_features_artifact = wu.first_used_artifact_of_type(
            log_run, MVFeaturesArtifact.wandb_artifact_type
        )
        return MVFeaturesArtifact.from_wandb_artifact(mv_features_artifact)

    def get_animation_from_lineage(self):
        mv_features_artifact = self.get_mv_features_from_lineage()
        return mv_features_artifact.get_animation_from_lineage()
