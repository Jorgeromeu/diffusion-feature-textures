from typing import Any, Callable, List

import codename
from omegaconf import DictConfig, OmegaConf

import wandb
from text3d2video.generative_rendering.configs import RunConfig
from text3d2video.omegaconf_util import dictconfig_equivalence
from wandb.apis.public import Run


def equivalent_configs(cfg1: DictConfig, cfg2: DictConfig) -> bool:
    cfg1_norun = cfg1.copy()
    cfg2_norun = cfg2.copy()

    # ignore run
    if "run" in cfg1:
        cfg1_norun.run = cfg1.run.copy()
    if "run" in cfg2:
        cfg2_norun.run = cfg2.run.copy()

    return dictconfig_equivalence(cfg1_norun, cfg2_norun)


class WandbExperiment:
    experiment_name: str
    run_fn: Callable[[DictConfig], Any]

    def run_configs(self) -> List[DictConfig]:
        """
        Return a list of run configs corresponding to the experiment
        """
        pass

    def execute_run(self, cfg: DictConfig, group: str):
        """
        Execute a run with given config, group and experiment name
        """
        run_config: RunConfig = cfg.run
        # set experiment tag
        if run_config.tags is None:
            run_config.tags = []
        run_config.tags.append(self.experiment_name)
        # set group
        run_config.group = group
        # execute
        self.run_fn(cfg)

    def execute_runs(self, group: str = None):
        if group is None:
            group = codename.codename(separator="-")
        print(f"Running {self.experiment_name} with group {group}")
        for cfg in self.run_configs():
            self.execute_run(cfg, group)

    def execute_runs_in_place(self, group: str = None):
        new_cfgs = self.run_configs()
        existing_runs = self.get_runs_in_group(group)
        existing_cfgs = [OmegaConf.create(r.config) for r in existing_runs]

        # delete existing runs that are not in the new configs
        delete_runs: List[Run] = []
        for run, cfg in zip(existing_runs, existing_cfgs):
            run_valid = any([equivalent_configs(cfg, e) for e in new_cfgs])
            if not run_valid:
                delete_runs.append(run)

        print(f"Deleting {len(delete_runs)} runs")
        for run in delete_runs:
            run.delete(delete_artifacts=True)

        # execute new runs that are not in the existing runs
        exec_configs = []
        for cfg in new_cfgs:
            run_exists = any([equivalent_configs(cfg, e) for e in existing_cfgs])
            if not run_exists:
                self.execute_run(cfg, group)

        print(f"Executing {len(exec_configs)} new runs")
        for cfg in exec_configs:
            self.execute_run(cfg, group)

    @classmethod
    def find_latest_group(cls) -> str:
        """
        Find the latest group name for this experiment
        """

        api = wandb.Api()
        project_name = "diffusion-3D-features"
        experient_filter = {"tags": cls.experiment_name}
        runs = api.runs(project_name, filters=experient_filter)
        runs = list(runs)
        if len(runs) == 0:
            return None
        latest_group = runs[-1].group
        return latest_group

    @classmethod
    def get_runs_in_group(cls, group: str = None) -> List[Run]:
        """
        Get all runs in a group
        """

        if group is None:
            group = cls.find_latest_group()

        api = wandb.Api()
        project_name = "diffusion-3D-features"
        filter = {"tags": cls.experiment_name, "group": group}
        runs = api.runs(project_name, filters=filter)
        runs = list(runs)
        return runs
