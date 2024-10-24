import logging
import shutil
import tempfile
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

import wandb
from wandb import Artifact


def setup_run(cfg: DictConfig):
    """
    Setup wandb run and log config
    """

    run_config = cfg.run
    wandb_mode = "online" if run_config.wandb else "disabled"

    wandb.init(
        project="diffusion-3d-features",
        job_type=run_config.job_type,
        mode=wandb_mode,
        tags=run_config.tags,
    )

    # add config
    wandb.config.update(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    do_run = True

    if run_config.instant_exit:
        print("Instant exit enabled")
        wandb.finish()
        do_run = False

    return do_run


def api_artifact(artifact_tag: str):
    """
    Get an artifact from the api
    """

    api = wandb.Api()
    return api.artifact(f"romeu/diffusion-3D-features/{artifact_tag}")


def wandb_is_enabled():
    return wandb.run is not None and not wandb.run.disabled


def get_artifact(artifact_tag: str):
    """
    If in run, use the artifact from the run, otherwise use the api
    """

    if wandb_is_enabled():
        return wandb.use_artifact(artifact_tag)

    return api_artifact(artifact_tag)


def first_logged_artifact_of_type(run, artifact_type: str) -> Artifact:
    for artifact in run.logged_artifacts():
        if artifact.type == artifact_type:
            return artifact
    return None


def first_used_artifact_of_type(run, artifact_type: str) -> Artifact:
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
    artifacts = api.artifacts(artifact_type, f"romeu/diffusion-3D-features/{artifact_name}")

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

    print(f"{artifact_name}: Deleting artifacts")
    collection.delete()


class ArtifactWrapper:
    """
    Wrapper over wandb artifact to ease reading/writing from artifacts
    """

    # the type id of the wandb artifact class
    wandb_artifact_type: str
    wandb_artifact: Artifact = None

    def __init__(self, folder: Path = None, artifact: Artifact = None):
        self.folder = folder
        self.wandb_artifact = artifact

    def setup_tempdir(self):
        self.folder = Path(tempfile.mkdtemp())
        logging.info(
            "Created %s artifact at %s",
            self.wandb_artifact.name,
            str(self.folder.absolute()),
        )

    def delete_folder(self):
        shutil.rmtree(self.folder)
        logging.info(
            "Deleted %s artifact at %s",
            self.wandb_artifact.name,
            str(self.folder.absolute()),
        )

    @classmethod
    def create_empty_artifact(cls, name: str):
        artifact = Artifact(name, type=cls.wandb_artifact_type)
        wrapper = cls(artifact=artifact)
        wrapper.setup_tempdir()
        return wrapper

    @classmethod
    def from_wandb_artifact(cls, artifact: Artifact, download=False):
        if download:
            artifact.download()

        # pylint: disable=protected-access
        folder = Path(artifact._default_root())
        wrapper = cls(folder=folder, artifact=artifact)
        return wrapper

    @classmethod
    def from_wandb_artifact_tag(cls, artifact_tag: str, download=False):
        artifact = get_artifact(artifact_tag)
        return cls.from_wandb_artifact(artifact, download)

    def log_if_enabled(self):
        self.wandb_artifact.add_dir(self.folder)

        if wandb_is_enabled():
            logging.info("Logging artifact %s", self.wandb_artifact.name)
            wandb.log_artifact(self.wandb_artifact)
            self.delete_folder()
            return

        logging.info(
            "Skippinglogging %s artifact at %s",
            self.wandb_artifact.name,
            str(self.folder.absolute()),
        )

    def logged_by(self):
        return self.wandb_artifact.logged_by()
