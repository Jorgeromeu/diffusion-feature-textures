from typing import Any, List

import tabulate
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

import wandb
import wandb_util.wandb_util as wbu
from text3d2video.utilities.omegaconf_util import (
    dictconfig_diff,
    dictconfig_flattened_keys,
)
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


def get_distinctive_keys(configs: List[DictConfig]) -> List[str]:
    keys = dictconfig_flattened_keys(configs[0])

    # find keys that are not the same for all runs
    distinctive_keys = []
    for key in keys:
        all_values = [OmegaConf.select(cfg, key) for cfg in configs]
        all_same = all([v == all_values[0] for v in all_values])
        if not all_same:
            distinctive_keys.append(key)

    return distinctive_keys


def run_config_distinctive_keys_table(configs: List[DictConfig]):
    keys = dictconfig_flattened_keys(configs[0])

    # find keys that are not the same for all runs
    distinctive_keys = []
    for key in keys:
        all_values = [OmegaConf.select(cfg, key) for cfg in configs]
        all_same = all([v == all_values[0] for v in all_values])
        if not all_same:
            distinctive_keys.append(key)

    # alphabetically sort
    distinctive_keys = sorted(distinctive_keys)

    # arrange as table
    rows = []
    rows.append(distinctive_keys)

    for cfg in configs:
        values = [OmegaConf.select(cfg, key) for key in distinctive_keys]
        rows.append(values)

    return tabulate.tabulate(rows, headers="firstrow")


def find_run_by_config(cfg: DictConfig) -> Run:
    """
    Find a run by its config
    """

    # get all runs
    api = wandb.Api()
    project_name = "diffusion-3D-features"
    runs = api.runs(project_name)
    runs = list(runs)

    for run in tqdm(runs):
        run_cfg = OmegaConf.create(run.config)
        if equivalent_configs(run_cfg, cfg):
            return run

    return None


def find_runs_by_configs(cfgs: list[DictConfig]) -> Run:
    """
    Find a run by its config
    """

    # get all runs
    api = wandb.Api()
    project_name = "diffusion-3D-features"
    runs = api.runs(project_name)
    runs = list(runs)

    found_runs = []

    for run in tqdm(runs):
        run_cfg = OmegaConf.create(run.config)

        for cfg in cfgs:
            is_equiv = equivalent_configs(run_cfg, cfg)
            if is_equiv:
                found_runs.append(run)

    return found_runs


def update_tags(run: Run, new_tags: List[str] = None, remove_tags: List[str] = None):
    """
    Update tags of a run
    """

    if new_tags is None:
        new_tags = []

    if remove_tags is None:
        remove_tags = []

    updated_tags = run.tags.copy()

    for tag in remove_tags:
        try:
            updated_tags.remove(tag)
        except ValueError:
            pass

    updated_tags.extend(new_tags)

    run.tags = updated_tags
    run.update()


def find_runs(tag: str = None) -> List[Run]:
    api = wandb.Api()
    project_name = "diffusion-3D-features"
    filter = {"tags": tag}
    runs = api.runs(project_name, filters=filter)
    runs = list(runs)
    return runs


def runs_table(
    runs: List[Run],
    show_url=False,
    show_state=True,
    show_distinctive_cofig=False,
):
    if len(runs) == 0:
        return

    fields = ["name"]
    if show_state:
        fields.append("state")
    if show_url:
        fields.append("url")

    configs = [OmegaConf.create(run.config) for run in runs]
    distinctive_keys = get_distinctive_keys(configs)

    rows = []
    for run in runs:
        row = []
        values = [getattr(run, field) for field in fields]
        row.extend(values)

        if show_distinctive_cofig:
            config = OmegaConf.create(run.config)
            distinctive_values = [
                OmegaConf.select(config, key) for key in distinctive_keys
            ]
            row.extend(distinctive_values)

        rows.append(row)

    headers = fields

    if show_distinctive_cofig:
        headers.extend(distinctive_keys)

    return print(tabulate.tabulate(rows, headers=headers))


class WandbExperiment:
    experiment_name: str
    run_fn: Any

    def run_configs(self) -> List[DictConfig]:
        """
        Return a list of run configs corresponding to the experiment
        """
        pass

    def print_distinctive_keys(self):
        configs = self.run_configs()

        # find all keys in the configs
        keys = dictconfig_flattened_keys(configs[0])

        # find keys that are not the same for all runs
        distinctive_keys = []
        for key in keys:
            all_values = [OmegaConf.select(cfg, key) for cfg in configs]
            all_same = all([v == all_values[0] for v in all_values])
            if not all_same:
                distinctive_keys.append(key)

        # alphabetically sort
        distinctive_keys = sorted(distinctive_keys)

        # arrange as table
        rows = []
        rows.append(distinctive_keys)

        for cfg in configs:
            values = [OmegaConf.select(cfg, key) for key in distinctive_keys]
            rows.append(values)

        print(tabulate.tabulate(rows, headers="firstrow"))

    def execute_run(self, cfg: DictConfig):
        """
        Execute a run with given config, in the experiment experiment
        """

        if wbu.wandb_is_enabled():
            wandb.finish()

        run_config: wbu.RunConfig = cfg.run

        # set experiment tag
        if run_config.tags is None:
            run_config.tags = []
        run_config.tags.append(self.experiment_name)

        # execute
        # TODO instantiate a job/process instead?
        self.run_fn(cfg)

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

    def get_runs_to_remove(self):
        # new runs, and existing runs
        new_cfgs = self.run_configs()
        existing_runs = self.get_logged_runs()

        # find runs to remove
        remove_runs = []

        # check all existing runs
        for existing_run in existing_runs:
            existing_cfg = OmegaConf.create(existing_run.config)
            in_new = False
            for new_cfg in new_cfgs:
                is_equiv = equivalent_configs(existing_cfg, new_cfg)
                in_new |= is_equiv
            if not in_new:
                remove_runs.append(existing_run)

        return remove_runs

    def get_configs_to_run(self):
        # new runs, and existing runs
        new_cfgs = self.run_configs()
        existing_runs = self.get_logged_runs()
        existing_cfgs = [OmegaConf.create(run.config) for run in existing_runs]

        # find runs to execute
        exec_configs = []
        for new_cfg in new_cfgs:
            in_existing = False
            for existing_cfg in existing_cfgs:
                is_equiv = equivalent_configs(existing_cfg, new_cfg)
                in_existing |= is_equiv
            if not in_existing:
                exec_configs.append(new_cfg)

        return exec_configs

    def execute_runs(self, dry_run=False, delete_existing=False):
        remove_runs = self.get_runs_to_remove()
        exec_configs = self.get_configs_to_run()

        if dry_run:
            remove_verb = "delete" if delete_existing else "remove"
            print(f"Would {remove_verb} {len(remove_runs)} runs")
            runs_table(remove_runs, show_distinctive_cofig=False)

            print(f"Would execute {len(exec_configs)} new runs")
            return

        remove_verb = "Deleting" if delete_existing else "Removing"
        print(f"{remove_verb} {len(remove_runs)} runs")
        for r in remove_runs:
            if delete_existing:
                r.delete()
            else:
                update_tags(r, remove_tags=[self.experiment_name])

        print(f"Executing {len(exec_configs)} new runs")
        for cfg in tqdm(exec_configs):
            self.execute_run(cfg)
