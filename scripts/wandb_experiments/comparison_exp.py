from itertools import product
from typing import Dict, List

from attr import dataclass
from omegaconf import DictConfig, OmegaConf

import wandb_util.wandb_util as wbu
from scripts.wandb_runs.run_generative_rendering import (
    run_generative_rendering,
)
from text3d2video.utilities.omegaconf_util import (
    omegaconf_from_dotdict,
)


@dataclass
class MultiDimSweepConfig:
    base_config: DictConfig
    override_dims: List[List[Dict]]
    override_dim_labels: List[List[str]]


def multidim_sweep_exp(config: MultiDimSweepConfig):
    # Generate all combinations of overrides
    overrides_flat = []
    for combo in product(*config.override_dims):
        merged = {}
        for d in combo:
            merged.update(d)
        overrides_flat.append(merged)

    # create all overrides
    override_dictconfigs = [omegaconf_from_dotdict(o) for o in overrides_flat]
    overriden_configs = [
        OmegaConf.merge(config.base_config, o) for o in override_dictconfigs
    ]

    runs = [
        wbu.RunSpec(f"o_{i}", run_generative_rendering, o)
        for i, o in enumerate(overriden_configs)
    ]

    return runs
