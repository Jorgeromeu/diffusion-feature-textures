
import shutil
import tempfile
from pathlib import Path

from wandb.apis.public import Run

from wandb import Artifact


def first_logged_artifact_of_type(run: Run, artifact_type: str) -> Artifact:
    for artifact in run.logged_artifacts():
        if artifact.type == artifact_type:
            return artifact
    return None


def first_used_artifact_of_type(run: Run, artifact_type: str) -> Artifact:
    for artifact in run.used_artifacts():
        if artifact.type == artifact_type:
            return artifact
    return None

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
