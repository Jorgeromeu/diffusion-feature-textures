from collections import defaultdict
from typing import Callable, Dict, List, Set

import torch
import torch.nn as nn
from torch import Tensor


class HookManager:
    """
    Utility class to manage hooks for a model
    """

    # keep track of named hooks
    _named_handles: Dict[str, torch.utils.hooks.RemovableHandle]

    def __init__(self) -> None:
        self._named_handles = dict()

    def named_hooks(self) -> Set[str]:
        return set(self._named_handles.keys())

    def add_named_hook(
        self,
        name: str,
        module: nn.Module,
        hook: Callable[[nn.Module, Tensor, Tensor], Tensor],
    ):
        """
        Create a named hook
        """

        # remove existing hook
        self.clear_named_hook(name)

        # register and save hook
        handle = module.register_forward_hook(hook)
        self._named_handles[name] = handle

    def clear_named_hook(self, name: str):
        if self._named_handles.get(name):
            self._named_handles[name].remove()


class DiffusionFeatureExtractor:

    # store saved features here, list because one per timestep
    _saved_features: Dict[str, List[torch.Tensor]]

    def __init__(self) -> None:
        self.hook_manager = HookManager()
        self._saved_features = defaultdict(lambda: [])

    def clear_features(self):
        self._saved_features = defaultdict(lambda: [])

    def n_saved_timesteps(self):
        first_key = list(self._saved_features.keys())[0]
        return len(self._saved_features[first_key])

    def add_save_feature_hook(self, name: str, module: nn.Module):
        self.hook_manager.add_named_hook(name, module, self._save_feature_hook(name))

    def get_feature(self, name: str, timestep=0):
        return self._saved_features[name][timestep]

    def _save_feature_hook(self, name: str):
        """
        Create a hook that saves the output of a module with key `name`
        """

        # pylint: disable=unused-argument
        def hook(module, inp, out):
            self._saved_features[name].append(out.cpu())

        return hook
