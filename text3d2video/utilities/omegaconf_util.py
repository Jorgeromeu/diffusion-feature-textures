from typing import Dict, List

from omegaconf import DictConfig, OmegaConf


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


def omegaconf_from_dotdict(d: dict):
    """
    Create a DictConfig from a dictionary with dot notation keys
    """

    dotlist = [f"{k}={v}" for k, v in d.items()]
    return OmegaConf.from_dotlist(dotlist)
