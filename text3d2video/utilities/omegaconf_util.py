from enum import Enum
from typing import Dict, List

from omegaconf import DictConfig, OmegaConf


def enums_to_values(cfg: DictConfig):
    for k, v in cfg.items():
        if isinstance(v, DictConfig):
            enums_to_values(v)
        elif isinstance(v, Enum):
            print(cfg[k])
            cfg[k] = None
            print(cfg[k])


def dictconfig_diff(conf1: DictConfig, conf2: DictConfig):
    diff = {}

    # Iterate through keys in conf1
    for key in conf1:
        if key not in conf2:
            diff[key] = {"only_in_conf1": conf1[key]}

        else:
            v1 = conf1[key]
            v2 = conf2[key]

            # read enums as values
            if isinstance(v1, Enum):
                v1 = v1.value
            if isinstance(v2, Enum):
                v2 = v2.value

            if v1 != v2:
                if isinstance(conf1[key], DictConfig) and isinstance(
                    conf2[key], DictConfig
                ):
                    nested_diff = dictconfig_diff(
                        conf1[key], conf2[key]
                    )  # Recurse for nested dictionaries
                    if nested_diff:
                        diff[key] = nested_diff
                else:
                    diff[key] = {"conf1": conf1[key], "conf2": conf2[key]}

    # Check for keys in conf2 that are not in conf1
    for key in conf2:
        if key not in conf1:
            diff[key] = {"only_in_conf2": conf2[key]}

    return diff


def dictconfig_flattened_keys(cfg: DictConfig):
    """
    Return a list of all keys to primitives in a DictConfig
    """

    keys = []
    for key in cfg:
        if isinstance(cfg[key], DictConfig):
            nested_keys = dictconfig_flattened_keys(cfg[key])
            for nested_key in nested_keys:
                keys.append(f"{key}.{nested_key}")
        else:
            keys.append(key)
    return keys


def dictconfig_equivalence(cfg1: DictConfig, cfg2: DictConfig) -> bool:
    """
    Check if two DictConfigs are equal
    """

    return not dictconfig_diff(cfg1, cfg2)


def get_nonequal_keys(configs: List[DictConfig]) -> List[str]:
    values = {}
    nonequal_keys = set()

    for cfg in configs:
        for key in dictconfig_flattened_keys(cfg):
            value = OmegaConf.select(cfg, key)
            if key not in values:
                values[key] = value
            if values[key] != value:
                nonequal_keys.add(key)

    return list(nonequal_keys)


def matches_override(cfg: DictConfig, o: Dict):
    for k, v in o.items():
        if not OmegaConf.select(cfg, k) == v:
            return False
    return True
