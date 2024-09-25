import shutil
import tempfile
from pathlib import Path

from wandb.apis.public import Run

import wandb
from wandb import Artifact


def api_artifact(artifact_tag: str):
    api = wandb.Api()
    return api.artifact(f'romeu/diffusion-3D-features/{artifact_tag}')
    

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


def cleanup_artifact_collection(
    api: wandb.Api,
    artifact_name: str,
    artifact_type: str,
    delete_log_runs: bool = True,
    keep_latest: bool = True,
):

    artifacts = api.artifacts(
        artifact_type, f"romeu/diffusion-3D-features/{artifact_name}"
    )

    if artifacts is None:
        print(f"{artifact_name}: No artifacts found")
        return

    print(f"{artifact_name}: Found {len(artifacts)} artifacts")

    for artifact in artifacts:

        # if keep latest ignore the run
        if keep_latest and "latest" in artifact.aliases:
            continue

        # delete the run that logged the artifact
        if delete_log_runs:
            logging_run = artifact.logged_by()
            logging_run.delete()

        # delete the artifact
        artifact.delete(delete_aliases=True)


def delete_artifact_collection(
    api: wandb.Api,
    artifact_name: str,
    artifact_type: str,
):
    collection = api.artifact_collection(
        artifact_type, f"romeu/diffusion-3D-features/{artifact_name}"
    )

    if collection is None:
        print(f"{artifact_name}: No artifacts found")
        return

    else:
        print(f"{artifact_name}: Deleting artifacts")
        collection.delete()


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
