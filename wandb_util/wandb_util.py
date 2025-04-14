import hashlib
import json
import logging
import multiprocessing as mp
import shutil
import tempfile
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import List, Optional

from attr import dataclass
from bidict import bidict
from moviepy.editor import ImageSequenceClip
from omegaconf import DictConfig, OmegaConf

import wandb
from wandb import Artifact
from wandb.apis.public import Run


def hash_config(cfg: DictConfig) -> str:
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.sha1(config_str.encode("utf-8")).hexdigest()


@dataclass
class RunConfig:
    wandb: bool = True
    download_artifacts: bool = False
    name: Optional[str] = None
    group: Optional[str] = None
    tags: list[str] = None

    def append_tags(self, tags: List[str]):
        if self.tags is None:
            self.tags = []

        self.tags += tags


# path to store local artifact data before logging to wandb
ARTIFACTS_LOCAL_PATH = "/tmp/local_artifacts/"


def setup_run(cfg: DictConfig, run_config: RunConfig, job_type: str) -> bool:
    """
    Setup wandb run and log its config
    """

    # get config dict
    config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    # init wandb
    wandb_mode = "online" if run_config.wandb else "disabled"

    # get tags
    tags = run_config.tags
    if tags is None:
        tags = []

    # add hash-tag
    hash = hash_config(cfg)
    hash_tag = f"hash:{hash}"
    tags.append(hash_tag)

    wandb.init(
        project="diffusion-3d-features",
        job_type=job_type,
        mode=wandb_mode,
        tags=tags,
        group=run_config.group,
        name=run_config.name,
    )

    # add hash to summary
    wandb.summary["hash"] = hash

    # log config
    wandb.config.update(config_dict)


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

    def log_standalone(self, aliases=None, delete_localfolder=False):
        wandb.init(project="diffusion-3D-features", job_type="log_artifact_standalone")
        self.log_if_enabled(aliases, delete_localfolder)
        wandb.finish()

    def logged_by(self):
        return self.wandb_artifact.logged_by()


class WandbRun:
    job_type: str

    def __init__(self):
        pass

    def _run(self, cfg: DictConfig):
        pass

    def execute(self, cfg: DictConfig, run_config: RunConfig):
        # init wandb
        setup_run(cfg, run_config, self.job_type)
        # run
        self._run(cfg)
        # finalize
        run = wandb.run
        wandb.finish()
        return run


class RunSpecification:
    run_fun: WandbRun
    config: DictConfig
    run_config: RunConfig
    depends_on: List

    def __init__(
        self,
        name: str,
        run_fun: WandbRun,
        config: DictConfig,
        run_config: RunConfig = None,
        depends_on=None,
    ):
        self.run_fun = run_fun
        self.config = config
        self.run_config = run_config
        self.depends_on = depends_on or []

        if self.run_config is None:
            self.run_config = RunConfig()

        self.run_config.name = name

    def __repr__(self):
        return f"{self.run_config.name}({self.depends_on})"

    def as_process(self):
        return mp.Process(
            target=self.run_fun.execute,
            args=(self.config, self.run_config),
        )


@dataclass
class ExperimentSyncAction:
    to_delete: List[Run]
    to_run: List[RunSpecification]


class Experiment:
    experiment_name: str
    config: DictConfig

    def __init__(
        self,
        group_name: str,
    ):
        self.group_name = group_name

    def specification(self) -> List[RunSpecification]:
        """
        Return a list of run descriptors corresponding to the experiment
        """
        pass

    def get_logged_runs(self) -> List[Run]:
        """
        Get all runs in the experiment
        """

        # create query
        query = {"$and": [{"tags": self.experiment_name}, {"group": self.group_name}]}

        api = wandb.Api()
        project_name = "diffusion-3D-features"
        runs = api.runs(project_name, filters=query)
        runs = list(runs)

        return runs

    def execute_runs(self, spec: List[RunSpecification]):
        """
        Execute a list of runs in the experiment
        """

        # add tags and group
        for run in spec:
            run.run_config.append_tags([self.experiment_name])
            run.run_config.group = self.group_name

        processes = [run.as_process() for run in spec]

        for p in processes:
            p.start()
            p.join()

    def calc_sync_experiment(self, rerun_all=False):
        """
        Obtain specification and logged runs and compute which runs in spec to execute, and which logged runs to delete so that the specification matches the logged runs
        """

        specification = self.specification()
        existing_runs = self.get_logged_runs()

        # if set, delete all existing, run all specified
        if rerun_all:
            return specification, existing_runs

        # 1. establish bidirectional mapping between specs and runs
        hash_to_spec = {hash_config(s.config): s for s in specification}
        spec_to_run = bidict()
        for r in existing_runs:
            hash = r.summary.get("hash")
            if hash in hash_to_spec and r.state == "finished":
                spec_to_run[hash_to_spec[hash]] = r

        # 2. find runs to run
        # run if:
        # - it has no corresponding existing run
        # - it depends on a run that has no corresponding existing run
        to_run = []
        for spec in specification:
            # if it has no corresponding existing run, run it
            existing_run = spec_to_run.get(spec)
            if existing_run is None:
                to_run.append(spec)

            for dep in spec.depends_on:
                # if it depends on a run that has no corresponding existing run, run it
                existing_dep = spec_to_run.get(dep)
                if existing_dep is None:
                    to_run.append(spec)

        # 3. find runs to delete
        # delete a run if:
        # - it has no corresponding spec
        # - it has a corresponding spec, but it is in to_run
        to_delete = []
        for r in existing_runs:
            spec = spec_to_run.inv.get(r)

            # if no spec, delete it
            if spec is None:
                to_delete.append(r)

            # if it has a spec, but the spec is in to_run, delete it
            elif spec in to_run:
                to_delete.append(r)

        # finally, toposort to_run
        to_run = topo_sort(to_run)

        return ExperimentSyncAction(
            to_run=to_run,
            to_delete=to_delete,
        )

    def sync_experiment(self, dry_run=False, rerun_all=False, interactive=True):
        action = self.calc_sync_experiment(rerun_all)

        if len(action.to_run) == 0 and len(action.to_delete) == 0:
            print("Experiment up-to-date!")
            return

        print(f"\nWould execute {len(action.to_run)} new runs:")
        for run in action.to_run:
            print(f"- {run.run_config.name}")

        print(f"\nWould delete {len(action.to_delete)} outdated runs:")
        for run in action.to_delete:
            print(f"- {run.name:<30} ({run.id})")
        print()

        if dry_run:
            print("Dry run, not executing")
            return

        if interactive:
            print("Do you want to continue? (y/n)")
            answer = input()
            if answer.lower() != "y":
                print("Aborting")
                return

        print(f"Deleting {len(action.to_delete)} runs")
        for run in action.to_delete:
            run.delete()

        print(f"Executing {len(action.to_run)} runs")

        self.execute_runs(action.to_run)

    def url(self):
        pass


def topo_sort(spec: List[RunSpecification]) -> List[RunSpecification]:
    """
    Topological sort of the runs in the experiment
    """

    # create graph
    graph = defaultdict(list)
    indegree = defaultdict(int)

    for run in spec:
        indegree[run] = 0

    for run in spec:
        for dep in run.depends_on:
            graph[dep].append(run)
            indegree[run] += 1

    # topological sort
    queue = [run for run in spec if indegree[run] == 0]
    sorted_spec = []

    while queue:
        run = queue.pop(0)
        sorted_spec.append(run)

        for dep in graph[run]:
            indegree[dep] -= 1
            if indegree[dep] == 0:
                queue.append(dep)

    return sorted_spec


def omegaconf_create_nested(d: dict):
    dotlist = [f"{k}={v}" for k, v in d.items()]
    return OmegaConf.from_dotlist(dotlist)


def wandb_run(job_type: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            print("wandb init")
            func(*args, **kwargs)
            print("wandb finish")

        return wrapper

    return decorator
