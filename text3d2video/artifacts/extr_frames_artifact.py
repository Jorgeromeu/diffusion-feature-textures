import re
from typing import Dict, List

import torch
from PIL import Image
from rerun import Tensor

import wandb_util.wandb_util as wbu


class ExtrFramesArtifact(wbu.ArtifactWrapper):
    wandb_artifact_type = "extr_frames"

    # writing
    def write(
        self, images: List[Image.Image], latents: Dict[int, Tensor], texture: Tensor
    ):
        for i, frame in enumerate(images):
            frame.save(self.folder / f"frame_{i}.png")
        torch.save(latents, self.folder / "latents.pt")
        torch.save(texture, self.folder / "texture.pt")

    # reading

    def read_frames(self) -> List[Image.Image]:
        def extract_index(p):
            match = re.search(r"frame_(\d+)\.png", p.name)
            return int(match.group(1)) if match else -1

        image_paths = sorted(self.folder.glob("frame_*.png"), key=extract_index)
        images = [Image.open(p) for p in image_paths]
        return images

    def read_latents(self) -> Dict[int, Tensor]:
        latents_path = self.folder / "latents.pt"
        return torch.load(latents_path)

    def read_texture(self) -> Tensor:
        texture_path = self.folder / "texture.pt"
        return torch.load(texture_path)
