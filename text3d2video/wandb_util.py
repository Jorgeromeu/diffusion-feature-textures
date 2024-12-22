import logging
import shutil
import tempfile
from pathlib import Path

import torch
from moviepy.editor import ImageSequenceClip
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

import wandb
from text3d2video.generative_rendering.configs import RunConfig
from wandb import Artifact

# path to store local artifact data before logging to wandb
ARTIFACTS_LOCAL_PATH = "/tmp/local_artifacts/"


def setup_run(run_config: RunConfig, cfg: DictConfig):
    """
    Setup wandb run and log its config
    """

    wandb_mode = "online" if run_config.wandb else "disabled"

    wandb.init(
        project="diffusion-3d-features",
        job_type=run_config.job_type,
        mode=wandb_mode,
        tags=run_config.tags,
        group=run_config.group,
    )

    # update run config config
    wandb.config.update(
        OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    )

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


def log_moviepy_clip(name, clip: ImageSequenceClip, fps=10):
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        temp_filename = f.name
        clip.write_videofile(temp_filename, codec="libx264", fps=fps)
        wandb.log({name: wandb.Video(temp_filename)})


class ArtifactWrapper:
    """
    Wrapper over wandb artifact to ease reading/writing from artifacts
    Holds reference to:
        - artifact local folder
        - wandb_artifact object
    """

    # the type id of the wandb artifact class
    wandb_artifact_type: str

    # artifact wrapper stores artifact and its local folder
    wandb_artifact: Artifact = None
    folder: Path

    def __init__(self, folder: Path = None, artifact: Artifact = None):
        self.folder = folder
        self.wandb_artifact = artifact

    def setup_localdir(self):
        artifact_name = self.wandb_artifact.name
        localdir_path = (
            Path(ARTIFACTS_LOCAL_PATH)
            / self.wandb_artifact_type
            / self.wandb_artifact.name
        )

        self.folder = localdir_path

        if self.folder.exists():
            shutil.rmtree(self.folder)
        self.folder.mkdir(exist_ok=True, parents=True)

        logging.info(
            "Created %s artifact at %s",
            artifact_name,
            str(self.folder.absolute()),
        )

    def _delete_localdir(self):
        shutil.rmtree(self.folder)
        logging.info(
            "Deleted %s local artifact folder at %s",
            self.wandb_artifact.name,
            str(self.folder.absolute()),
        )

    # Construcors

    @classmethod
    def create_empty_artifact(cls, name: str):
        """
        Create artifact, and initialize empty directory
        """
        artifact = Artifact(name, type=cls.wandb_artifact_type)
        wrapper = cls(artifact=artifact)
        wrapper.setup_localdir()
        return wrapper

    @classmethod
    def create_empty_artifact_from_folder(cls, folder: Path, name: str):
        """
        Create artifact, initialize from local folder
        """
        artifact = Artifact(name, type=cls.wandb_artifact_type)
        return cls(folder=folder, artifact=artifact)

    @classmethod
    def from_wandb_artifact(cls, artifact: Artifact, download=False):
        """
        Create from logged artifact
        """
        # pylint: disable=protected-access
        downloaded_folder = Path(artifact._default_root())

        # if folder not present download
        if not downloaded_folder.exists():
            artifact.download()

        # if download flag present, force redownload
        if download:
            artifact.download()

        # set up artifact wrapper
        wrapper = cls(folder=downloaded_folder, artifact=artifact)
        return wrapper

    @classmethod
    def from_wandb_artifact_tag(cls, artifact_tag: str, download=False):
        """
        Create from logged artifact tag
        """
        artifact = get_artifact(artifact_tag)
        return cls.from_wandb_artifact(artifact, download)

    def log_if_enabled(self, aliases=None, delete_localfolder=False):
        if aliases is None:
            aliases = []

        # add directory to artifact
        self.wandb_artifact.add_dir(self.folder)

        if wandb_is_enabled():
            logging.info("Logging artifact %s", self.wandb_artifact.name)
            wandb.log_artifact(self.wandb_artifact, aliases)

            if delete_localfolder:
                self._delete_localdir()
        else:
            logging.info(
                "Skipping logging %s artifact at %s",
                self.wandb_artifact.name,
                str(self.folder.absolute()),
            )

    def logged_by(self):
        return self.wandb_artifact.logged_by()


class SimpleArtifact(ArtifactWrapper):
    """
    Minimal example for an artifact class
    """

    wandb_artifact_type = "simple"

    def write_tensor(self, data: Tensor):
        torch.save(data, self.folder / "data.pt")

    def read_tensor(self):
        return torch.load(self.folder / "data.pt")
