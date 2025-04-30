import torch
import torchvision.transforms.functional as TF

import wandb_util.wandb_util as wbu
from text3d2video.util import hwc_to_chw


class TextureArtifact(wbu.ArtifactWrapper):
    wandb_artifact_type = "rgb_texture"

    def _texture_path(self):
        return self.folder / "texture.pt"

    def write_texture(self, texture: torch.Tensor):
        torch.save(texture.cpu(), self._texture_path())

    def read_texture(self, device="cuda") -> torch.Tensor:
        texture = torch.load(self._texture_path()).to(device)
        return texture

    def read_texture_pil(self):
        texture = self.read_texture("cpu")
        texture = hwc_to_chw(texture)
        return TF.to_pil_image(texture)
