from collections import defaultdict
from diffusers import UNet2DConditionModel
from typing import Callable, Dict, List, Set, Tuple
import numpy as np
import torch
from diffusers import DiffusionPipeline
from torch.utils.hooks import RemovableHandle
import torch.nn as nn
from torch import Tensor

class SDFeatureExtractor:

    # store saved features here, list because one per timestep
    _saved_features: Dict[int, list[np.array]]
    _handles: List[torch.utils.hooks.RemovableHandle]

    def __init__(self, pipe: DiffusionPipeline):
        self.unet = pipe.unet
        self._saved_features = dict()
        self.create_hooks()

    def create_hooks(self):
        for level, up_block in enumerate(self.unet.up_blocks): 
            self._saved_features[level] = []
            handle = up_block.register_forward_hook(self._save_feature_hook(level))

    def get_feature(self, level=0, timestep=0):
        return self._saved_features[level][timestep]

    def _save_feature_hook(self, level):
        def hook(module, inp, out):
            out_np = out.cpu().numpy()[0]
            self._saved_features[level].append(out_np)
        return hook


class FeatureExtractor:

    """
    Utility class to add named hooks to a model.
    """

    # keep track of named hooks
    _named_handles: dict[str, torch.utils.hooks.RemovableHandle]

    # store saved features here, list because one per timestep
    _saved_features: dict[str, List[torch.Tensor]]

    def __init__(self) -> None:
        self._named_handles = dict()
        self._saved_features = defaultdict(lambda: [])

    def add_named_hook(
        self,
        name: str,
        module: nn.Module,
        hook: Callable[[nn.Module, Tensor, Tensor], Tensor]
    ):

        # remove existing save_hook
        if self._named_handles.get(name):
            self._named_handles[name].remove()

        handle = module.register_forward_hook(hook)
        self._named_handles[name] = handle

    def add_save_named_feature_hook(self, name: str, module: nn.Module):
        self.add_named_hook(name, module, self.save_named_module_out_hook(name))

    def save_named_module_out_hook(self, name: str):

        """
        Create a hook that saves the output of a module with key `name`
        """
        def hook(module, inp, out):
            self._saved_features[name].append(out.cpu().numpy())
        return hook

    def clear_named_hook(self, name: str):
        if self._named_handles.get(name):
            self._named_handles[name].remove()

