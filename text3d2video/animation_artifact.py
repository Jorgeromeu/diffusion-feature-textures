import shutil
import tempfile
from pathlib import Path

from pytorch3d.structures import Meshes

from text3d2video.obj_io import load_objs_as_meshes
from text3d2video.util import ordered_sample
from wandb import Artifact


class ArtifactWrapper:

    artifact_type: str
    wandb_artifact: Artifact = None

    def __init__(self, folder: Path):
        self.folder = folder

    @classmethod
    def from_path(cls, path: Path):
        return cls(path)

    @classmethod
    def from_wandb_artifact(cls, artifact: Artifact):
        folder = Path(artifact.download())
        wrapper = cls(folder)
        wrapper.wandb_artifact = artifact
        return wrapper

    @staticmethod
    def write_to_path(folder: Path, **kwargs):
        pass

    @classmethod
    def create_wandb_artifact(cls, name: str, **kwargs) -> Artifact:

        # create temporary directory and write to it
        tempdir = tempfile.mkdtemp()
        cls.write_to_path(Path(tempdir), **kwargs)

        # create artifact and add directory
        artifact = Artifact(name, type=cls.artifact_type)
        artifact.add_dir(tempdir)

        # remove temporary directory
        shutil.rmtree(tempdir)

        return artifact


class AnimationArtifact(ArtifactWrapper):

    artifact_type = "animation"

    @staticmethod
    def write_to_path(folder: Path, animation_path: str, static_path: str):

        # copy static mesh
        shutil.copy(static_path, folder / "static.obj")

        # copy frames
        animation_dir = folder / "animation"
        animation_dir.mkdir()
        for frame in Path(animation_path).iterdir():
            number = frame.stem[-4:]
            frame_name = f"animation{number}.obj"
            shutil.copy(frame, animation_dir / frame_name)

    def get_static_mesh_path(self) -> Path:
        return self.folder / "static.obj"

    def get_frame_path(self, frame=1) -> Path:
        return self.folder / "animation" / f"animation{frame:04}.obj"

    def frame_nums(self, sample_n=None):

        frame_paths = (self.folder / "animation").iterdir()
        frame_nums = [int(path.stem[-4:]) for path in frame_paths]
        frame_nums = sorted(frame_nums)

        if sample_n is not None:
            frame_nums = ordered_sample(frame_nums, sample_n)

        return frame_nums
    
    def load_static_mesh(self, device: str = 'cuda') -> Meshes:
        return load_objs_as_meshes([self.get_static_mesh_path()], device=device)
    
    def load_frame(self, frame: int, device: str = 'cuda') -> Meshes:
        return load_objs_as_meshes([self.get_frame_path(frame)], device=device)
