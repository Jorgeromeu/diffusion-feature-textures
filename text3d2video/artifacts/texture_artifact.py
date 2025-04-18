import torch

import wandb_util.wandb_util as wbu


class TextureArtifact(wbu.ArtifactWrapper):
    wandb_artifact_type = "rgb_texture"

    def write_texture(self, texture: torch.Tensor):
        torch.save(texture.cpu(), self.folder / "texture.pt")

    def read_texture(self, device="cuda") -> torch.Tensor:
        texture_path = self.folder / "texture.pt"
        texture = torch.load(texture_path).to(device)
        return texture
