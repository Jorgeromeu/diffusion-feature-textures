import re
from typing import List

from PIL import Image

import wandb_util.wandb_util as wbu


class VideoArtifact(wbu.ArtifactWrapper):
    wandb_artifact_type = "video"

    # writing

    def write_frames(self, frames: List[Image.Image]):
        for i, frame in enumerate(frames):
            frame.save(self.folder / f"frame_{i}.png")

    # reading

    def read_frames(self) -> List[Image.Image]:
        def extract_index(p):
            match = re.search(r"frame_(\d+)\.png", p.name)
            return int(match.group(1)) if match else -1

        image_paths = sorted(self.folder.glob("frame_*.png"), key=extract_index)
        images = [Image.open(p) for p in image_paths]
        return images
