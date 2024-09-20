from pathlib import Path
import os

from pytorch3d.io import load_objs_as_meshes
from text3d2video.util import ordered_sample


class OBJAnimation:

    """
    Utility class for reading animations from a directory of .obj files
    """

    objs_dir: Path

    def __init__(self, objs_dir: Path, anim_name: str) -> None:
        self.objs_dir = objs_dir
        self.anim_name = anim_name

    def load_frame(self, frame=0, device='cuda'):
        path = self.objs_dir / f'{self.anim_name}{frame:04}.obj'
        mesh = load_objs_as_meshes([path], device=device)
        return mesh

    def framenums(self, sample_n=None):
        filenames = os.listdir(self.objs_dir)
        framenums = [int(name[-8:-4]) for name in filenames]

        if sample_n is not None:
            framenums = ordered_sample(framenums, sample_n)

        framenums = sorted(framenums)
        return framenums

    def num_frames(self):
        return max(self.framenums())
