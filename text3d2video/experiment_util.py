from typing import Any, Callable, List

from omegaconf import DictConfig, OmegaConf

import wandb
from text3d2video.generative_rendering.configs import RunConfig
from text3d2video.omegaconf_util import dictconfig_diff, dictconfig_equivalence
from wandb.apis.public import Run


def object_to_instantiate_config(obj: Any) -> DictConfig:
    """
    Convert an object to a DictConfig that can be used to instantiate the object
    """
    class_path = f"{obj.__module__}.{obj.__class__.__name__}"
    config = {"_target_": class_path}
    for key, value in obj.__dict__.items():
        config[key] = value
    return OmegaConf.create(config)


def configs_diff(cfg1: DictConfig, cfg2: DictConfig) -> bool:
    cfg1_norun = cfg1.copy()
    cfg2_norun = cfg2.copy()

    if "run" in cfg1:
        del cfg1_norun.run
    if "run" in cfg2:
        del cfg2_norun.run

    return dictconfig_diff(cfg1_norun, cfg2_norun)


def equivalent_configs(cfg1: DictConfig, cfg2: DictConfig) -> bool:
    return len(configs_diff(cfg1, cfg2)) == 0


class WandbExperiment:
    experiment_name: str
    run_fn: Callable[[DictConfig], Any]

    def run_configs(self) -> List[DictConfig]:
        """
        Return a list of run configs corresponding to the experiment
        """
        pass

    def _execute_run(self, cfg: DictConfig, group: str):
        """
        Execute a run with given config, group and experiment name
        """

        # TODO execute each run as a Job rather than calling a function

        run_config: RunConfig = cfg.run
        # set experiment tag
        if run_config.tags is None:
            run_config.tags = []
        run_config.tags.append(self.experiment_name)
        # set group
        run_config.group = group
        # execute
        self.run_fn(cfg)

    def execute_runs(self, group: str = None, dry_run=False, force_rerun=False):
        # new runs, and existing runs
        new_cfgs = self.run_configs()
        existing_runs = self.get_runs_in_group(group)
        existing_cfgs = [OmegaConf.create(run.config) for run in existing_runs]

        # find runs to delete
        del_runs = []
        keep_runs = []
        for existing_run in existing_runs:
            existing_cfg = OmegaConf.create(existing_run.config)
            in_new = False
            for new_cfg in new_cfgs:
                is_equiv = equivalent_configs(existing_cfg, new_cfg)
                in_new |= is_equiv
            if not in_new:
                del_runs.append(existing_run)
            else:
                keep_runs.append(existing_run)

        # find runs to execute
        exec_configs = []
        for new_cfg in new_cfgs:
            in_existing = False
            for existing_cfg in existing_cfgs:
                is_equiv = equivalent_configs(existing_cfg, new_cfg)
                in_existing |= is_equiv
            if not in_existing:
                exec_configs.append(new_cfg)

        if force_rerun:
            del_runs.extend(keep_runs)
            keep_configs = [OmegaConf.create(run.config) for run in keep_runs]
            exec_configs.extend(keep_configs)

        if dry_run:
            print(f"Would delete {len(del_runs)} runs")
            print(f"Would execute {len(exec_configs)} new runs")
            return

        print(f"Deleting {len(del_runs)} runs")
        for r in del_runs:
            r.delete(delete_artifacts=True)

        print(f"Executing {len(exec_configs)} new runs")
        for cfg in exec_configs:
            self._execute_run(cfg, group)

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
