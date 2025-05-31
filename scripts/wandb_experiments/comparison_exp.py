from itertools import product
from typing import Dict, List

from attr import dataclass
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

import wandb_util.wandb_util as wbu
from text3d2video.utilities.omegaconf_util import (
    omegaconf_from_dotdict,
)


@dataclass
class MultiDimSweepConfig:
    fun_path: str
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

    # overriden labels
    override_labels = list(product(*config.override_dim_labels))
    override_labels = [" | ".join(label) for label in override_labels]

    fun = get_method(config.fun_path)

    runs = [
        wbu.RunSpec(lbl, fun, o) for lbl, o in zip(override_labels, overriden_configs)
    ]

    return runs
