from pathlib import Path

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes

def load_frame_obj(objs_dir: Path, frame = 0, device='cuda') -> Meshes:

    directory_name = objs_dir.stem
    path = objs_dir / f'{directory_name}{frame:04}.obj'

    mesh = load_objs_as_meshes([path], device=device)
    return mesh