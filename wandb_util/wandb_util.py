import hashlib
import json
import logging
import multiprocessing as mp
import shutil
import tempfile
import time
from functools import wraps
from pathlib import Path
from typing import Callable, Dict, List, Optional

import networkx as nx
from attr import dataclass
from moviepy.editor import ImageSequenceClip
from omegaconf import DictConfig, OmegaConf

import wandb
from wandb import Artifact


def hash_dictconfig(cfg: DictConfig) -> str:
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
PROJECT_NAME = "diffusion-3D-features"


def wandb_is_enabled():
    return wandb.run is not None and not wandb.run.disabled


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
    hash = hash_dictconfig(cfg)
    hash_tag = f"hash:{hash}"
    tags.append(hash_tag)

    wandb.init(
        project=PROJECT_NAME,
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
    return api.artifact(f"romeu/{PROJECT_NAME}/{artifact_tag}")


def api_runs(query: Dict):
    api = wandb.Api()
    runs = api.runs(PROJECT_NAME, filters=query)
    runs = list(runs)
    return runs


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
        wandb.init(project=PROJECT_NAME, job_type="log_artifact_standalone")
        self.log_if_enabled(aliases, delete_localfolder)
        wandb.finish()

    def logged_by(self):
        return self.wandb_artifact.logged_by()


class RunSpec:
    run_fun: Callable
    config: DictConfig
    run_config: RunConfig
    depends_on: List

    def __init__(
        self,
        name: str,
        run_fun: Callable,
        config: DictConfig,
        run_config: RunConfig = None,
        depends_on=None,
    ):
        self.name = name
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
            target=self.run_fun,
            args=(self.config, self.run_config),
        )

    def launch(self):
        p = self.as_process()
        p.start()


@dataclass
class ExperimentSyncAction:
    to_delete: List
    to_run: List


def wandb_run(job_type: str):
    def decorator(func):
        @wraps(func)
        def wrapper(cfg: DictConfig, run_config: RunConfig, *args, **kwargs):
            # setup wandb run
            setup_run(cfg, run_config, job_type)
            # logic
            result = func(cfg, run_config, *args, **kwargs)
            # finalize wandb run
            wandb.finish()
            return result

        return wrapper

    return decorator


@wandb_run("experiment_config")
def experiment_run(cfg: DictConfig, run_config: RunConfig):
    pass


def calc_sync_experiment_algo(
    specification: nx.DiGraph, existing: List, hash_spec, hash_run
):
    run_hashes = set(hash_run(r) for r in existing)
    spec_hashes = set(hash_spec(s) for s in specification.nodes)

    # find specs that are not in existing
    to_run = set()
    for s in specification.nodes:
        spec_hash = hash_spec(s)
        if spec_hash not in run_hashes:
            to_run.add(s)

    # recursively find specs that depend on the ones we found
    to_run_descendants = set()
    for node in to_run:
        descendants = nx.descendants(specification, node)
        to_run_descendants |= descendants

    to_run |= to_run_descendants

    # toposort to_run
    to_run_subgraph = specification.subgraph(to_run)
    to_run_sorted = list(nx.topological_sort(to_run_subgraph))

    # delete run if:
    # - it is not in specification
    # - it is in to_run
    # - it is duplicate

    to_run_hashes = set(hash_spec(s) for s in to_run)
    observed_hashes = set()

    to_del = set()
    for e in existing:
        hash = hash_run(e)

        if hash in to_run_hashes:
            to_del.add(e)

        if hash not in spec_hashes:
            to_del.add(e)

        if hash in observed_hashes:
            to_del.add(e)

        observed_hashes.add(hash)

    return ExperimentSyncAction(to_run=to_run_sorted, to_delete=to_del)


def get_logged_runs(name: str, include_exp_run=False):
    # create query
    query = {"group": name}

    if not include_exp_run:
        query["tags"] = {"$nin": ["exp_run"]}

    return api_runs(query)


def calc_sync_experiment(
    exp_spec: Callable, exp_cfg: DictConfig, name: str, rerun_all=False
):
    exp_run = RunSpec("exp", experiment_run, exp_cfg, RunConfig(tags=["exp_run"]))
    specification = exp_spec(exp_cfg) + [exp_run]
    existing_runs = get_logged_runs(name, include_exp_run=True)

    if rerun_all:
        # rerun all runs
        return ExperimentSyncAction(
            to_run=specification,
            to_delete=existing_runs,
        )

    def hash_run(run):
        return run.summary.get("hash")

    def hash_spec(spec):
        return hash_dictconfig(spec.config)

    # make specification graph
    spec_graph = nx.DiGraph()
    for run in specification:
        spec_graph.add_node(run)
        for dep in run.depends_on:
            spec_graph.add_edge(dep, run)

    return calc_sync_experiment_algo(spec_graph, existing_runs, hash_spec, hash_run)


def sync_experiment(
    exp_fun, config, name, dry_run=False, rerun_all=False, interactive=True
):
    print(f"Experiment: {get_exp_url(name)}")

    action = calc_sync_experiment(exp_fun, config, name, rerun_all)

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
        time.sleep(0.5)
        print("Do you want to continue? (y/n)")
        answer = input()
        if answer.lower() != "y":
            print("Aborting")
            return

    print(f"Deleting {len(action.to_delete)} runs")
    for run in action.to_delete:
        run.delete()

    print(f"Executing {len(action.to_run)} runs")

    # execute the runs
    # add tags and group
    for run in action.to_run:
        run.run_config.group = name

    processes = [run.as_process() for run in action.to_run]

    for p in processes:
        p.start()
        p.join()


def get_exp_config(name):
    query = {
        "group": name,
        "tags": {"$in": ["exp_run"]},
    }
    exp_run = api_runs(query)[0]
    exp_config = OmegaConf.create(exp_run.config)
    return exp_config


def get_exp_url(name: str):
    return f"https://wandb.ai/romeu/diffusion-3D-features/groups/{name}/workspace"
