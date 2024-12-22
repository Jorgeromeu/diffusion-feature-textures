from enum import Enum

from omegaconf import DictConfig


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


def dictconfig_equivalence(cfg1: DictConfig, cfg2: DictConfig) -> bool:
    """
    Check if two DictConfigs are equal
    """

    return not dictconfig_diff(cfg1, cfg2)
