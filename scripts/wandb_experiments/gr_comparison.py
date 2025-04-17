from itertools import product
from typing import Dict, List

import numpy as np
from attr import dataclass
from omegaconf import DictConfig, OmegaConf

import wandb_util.wandb_util as wbu
from scripts.wandb_runs.run_generative_rendering import (
    run_generative_rendering,
)
from text3d2video.utilities.omegaconf_util import (
    matches_override,
    omegaconf_from_dotdict,
)


@dataclass
class ComparisonExperimentConfig:
    base_config: DictConfig
    override_dims: List[List[Dict]]
    override_dim_labels: List[List[str]]


class ComparisonExperiment(wbu.Experiment):
    experiment_name = "gr_comparison"
    config: ComparisonExperimentConfig

    def specification(self):
        # Generate all combinations of overrides
        overrides_flat = []
        for combo in product(*self.config.override_dims):
            merged = {}
            for d in combo:
                merged.update(d)
            overrides_flat.append(merged)

        # create all overrides
        override_dictconfigs = [omegaconf_from_dotdict(o) for o in overrides_flat]
        overriden_configs = [
            OmegaConf.merge(self.config.base_config, o) for o in override_dictconfigs
        ]

        runs = [
            wbu.RunSpec(f"o_{i}", run_generative_rendering, o)
            for i, o in enumerate(overriden_configs)
        ]

        return runs

    def get_runs_grouped(self):
        # get the overrides from the experiment run
        exp_config = self.get_exp_config()
        override_dims = exp_config.override_dims

        shape = [len(dim) for dim in override_dims]
        grid = np.empty(shape, dtype=object)

        # for each run, check where it is in the grid
        for run in self.get_logged_runs():
            config = OmegaConf.create(run.config)

            # for each dimension, get index where overrides match
            index = []
            for dim, overrides in enumerate(override_dims):
                for i, o in enumerate(overrides):
                    if matches_override(config, o):
                        index.append(i)

            # add the run to the grid
            grid[tuple(index)] = run

        return grid
