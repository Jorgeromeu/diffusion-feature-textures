import logging
import multiprocessing as mp
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional

import torch
from attr import dataclass
from moviepy.editor import ImageSequenceClip
from omegaconf import DictConfig, OmegaConf
from torch import Tensor

import wandb
from wandb import Artifact
from wandb.apis.public import Run


@dataclass
class RunConfig:
    wandb: bool
    instant_exit: bool
    download_artifacts: bool
    name: Optional[str]
    group: Optional[str]
    tags: list[str]


# path to store local artifact data before logging to wandb
ARTIFACTS_LOCAL_PATH = "/tmp/local_artifacts/"


def setup_run(cfg: DictConfig, job_type: str) -> bool:
    """
    Setup wandb run and log its config
    """

    run_config = cfg.run

    wandb_mode = "online" if run_config.wandb else "disabled"

    wandb.init(
        project="diffusion-3d-features",
        job_type=job_type,
        mode=wandb_mode,
        tags=run_config.tags,
        group=run_config.group,
        name=run_config.name,
    )

    # log run config
    config = cfg.copy()
    del config.run
    config_dict = OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    wandb.config.update(config_dict)

    do_run = True
    if run_config.instant_exit:
        print("Instant exit enabled")
        wandb.finish()
        do_run = False

    return do_run


def api_artifact(artifact_tag: str) -> Artifact:
    """
    Get an artifact from the api
    """

    api = wandb.Api()
    return api.artifact(f"romeu/diffusion-3D-features/{artifact_tag}")


def wandb_is_enabled():
    return wandb.run is not None and not wandb.run.disabled


def resolve_artifact_tag(artifact_tag: str):
    name, _ = artifact_tag.split(":")
    art = api_artifact(artifact_tag)
    true_version = art.version
    return f"{name}:{true_version}"


def get_artifact(artifact_tag: str):
    """
    If in run, use the artifact from the run, otherwise use the api
    """

    if wandb_is_enabled():
        return wandb.use_artifact(artifact_tag)

    return api_artifact(artifact_tag)


def first_logged_artifact_of_type(
    run, artifact_type: str, name: str = None
) -> Artifact:
    for artifact in run.logged_artifacts():
        if artifact.type == artifact_type:
            if name is None or artifact.name.startswith(name):
                return artifact

            return artifact
    return None


def first_used_artifact_of_type(run, artifact_type: str) -> Artifact:
    for artifact in run.used_artifacts():
        if artifact.type == artifact_type:
            return artifact
    return None


def logged_artifacts(run, type=None, name_startswith=None):
    artifacts = []
    for art in run.logged_artifacts():
        if type is not None and art.type != type:
            continue

        if name_startswith is not None and not art.name.startswith(name_startswith):
            continue

        artifacts.append(art)

    return artifacts


def used_artifacts(run, type=None, name_startswith=None):
    artifacts = []
    for art in run.used_artifacts():
        if type is not None and art.type != type:
            continue

        if name_startswith is not None and not art.name.startswith(name_startswith):
            continue

        artifacts.append(art)

    return artifacts


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

    """
    Specify the artifact type to classify in wandb
    """

    wandb_artifact_type = "simple"

    """
    Implement methods which read and write to self.folder
    """

    def write_tensor(self, data: Tensor):
        torch.save(data, self.folder / "data.pt")

    def read_tensor(self):
        return torch.load(self.folder / "data.pt")


class WandbRun:
    job_type: str

    def __init__(self):
        pass

    def _run(self, cfg: DictConfig):
        pass

    def execute(self, cfg: DictConfig):
        # init wandb
        do_run = setup_run(cfg, self.job_type)
        if not do_run:
            return

        self._run(cfg)
        run = wandb.run
        wandb.finish()
        return run


@dataclass
class RunDescriptor:
    run_fun: WandbRun
    config: DictConfig

    def append_tags(self, tags: List[str]):
        self.config.run.tags += tags

    def as_process(self):
        return mp.Process(
            target=self.run_fun.execute,
            args=(self.config,),
        )


def spec_matches_run(spec: RunDescriptor, run: Run):
    same_job_type = spec.run_fun.job_type == run.job_type

    if not same_job_type:
        return False

    spec_config = spec.config.copy()
    del spec_config.run
    spec_config_dict = OmegaConf.to_container(
        spec_config, resolve=True, throw_on_missing=True
    )

    return spec_config_dict == run.config


class Experiment:
    experiment_name: str
    config: DictConfig

    def specification(self) -> List[RunDescriptor]:
        """
        Return a list of run descriptors corresponding to the experiment
        """
        pass

    def execute_runs(self, dry_run=False):
        mp.set_start_method("spawn")
        runs = self.specification()

        if dry_run:
            print(f"would execute {len(runs)} runs")
            return

        for run in runs:
            run.append_tags([self.experiment_name])

        processes = [run.as_process() for run in runs]

        for p in processes:
            p.start()
            p.join()

    def get_logged_runs(cls, filters=None) -> List[Run]:
        """
        Get all runs in the experiment
        """

        run_filters = {"tags": cls.experiment_name}
        if filters is not None:
            run_filters = {**run_filters, **filters}

        api = wandb.Api()
        project_name = "diffusion-3D-features"
        runs = api.runs(project_name, filters=run_filters)
        runs = list(runs)

        return runs

    def sync_experiment(self, dry_run=False):
        specification = self.specification()
        existing_runs = self.get_logged_runs()

        # find runs remaining to run
        to_run = []
        for spec in specification:
            run_exists = any([spec_matches_run(spec, run) for run in existing_runs])
            if not run_exists:
                to_run.append(spec)

        # find runs to delete
        to_del = []
        for run in existing_runs:
            in_spac = any([spec_matches_run(spec, run) for spec in specification])
            if not in_spac:
                to_del.append(run)

        if dry_run:
            print(f"would execute {len(to_run)} runs")
            print(f"would delete {len(to_del)} runs")
            return

        print(f"Deleting {len(to_del)} runs")
        for run in to_del:
            run.delete()

        print(f"Executing {len(to_run)} runs")
        mp.set_start_method("spawn")

        for run in to_run:
            run.append_tags([self.experiment_name])

        processes = [run.as_process() for run in to_run]

        for p in processes:
            p.start()
            p.join()
