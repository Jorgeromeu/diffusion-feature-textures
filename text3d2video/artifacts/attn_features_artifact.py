from typing import List

import torchvision.transforms.functional as TF
from PIL.Image import Image
from torchvision.io import read_image

import text3d2video.wandb_util as wbu
from text3d2video.disk_multidict import TensorDiskMultiDict


class AttentionFeaturesArtifact(wbu.ArtifactWrapper):
    wandb_artifact_type = "attn_data"

    def _get_features_path(self):
        return self.folder / "features"

    # writing

    def create_features_diskdict(self) -> TensorDiskMultiDict:
        self._get_features_path().mkdir(exist_ok=True)
        return TensorDiskMultiDict(self._get_features_path())

    def write_images(self, images: List[Image], fps=10):
        for i, img in enumerate(images):
            img.save(self.folder / f"image_{i}.png")

    # reading

    def get_features_diskdict(self) -> TensorDiskMultiDict:
        return TensorDiskMultiDict(self._get_features_path())

    def get_image(self, i: int) -> Image:
        return TF.to_pil_image(read_image(self.folder / f"image_{i}.png"))

    def get_images(self):
        img_paths = []
        for img in self.folder.iterdir():
            if img.suffix == ".png":
                img_paths.append(img)
        img_paths.sort()

        ims = []
        for path in img_paths:
            ims.append(TF.to_pil_image(read_image(path)))
        return ims
