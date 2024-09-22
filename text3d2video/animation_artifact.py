import shutil
import tempfile
from pathlib import Path

from text3d2video.util import ordered_sample
from wandb import Artifact


class ArtifactWrapper:

    artifact_type: str

    def __init__(self, folder: Path):
        self.folder = folder

    @classmethod
    def from_path(cls, path: Path):
        return cls(path)

    @classmethod
    def from_wandb_artifact(cls, artifact: Artifact):
        folder = Path(artifact.download())
        return cls(folder)

    @staticmethod
    def write_to_path(dir: Path, **kwargs):
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
    def write_to_path(dir: Path, animation_path: str, static_path: str):

        # copy static mesh
        shutil.copy(static_path, dir / "static.obj")

        # copy frames
        animation_dir = dir / "animation"
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
