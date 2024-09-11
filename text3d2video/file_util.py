from pathlib import Path
import os
import re

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes

from text3d2video.util import ordered_sample

class OBJAnimation:

    objs_dir: Path

    def __init__(self, objs_dir: Path) -> None:
        self.objs_dir = objs_dir

    def load_frame(self, frame=0, device='cuda'):
        directory_name = self.objs_dir.stem
        path = self.objs_dir / f'{directory_name}{frame:04}.obj'
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

def load_frame_obj(objs_dir: Path, frame=0, device='cuda') -> Meshes:

    directory_name = objs_dir.stem
    path = objs_dir / f'{directory_name}{frame:04}.obj'

    mesh = load_objs_as_meshes([path], device=device)
    return mesh
