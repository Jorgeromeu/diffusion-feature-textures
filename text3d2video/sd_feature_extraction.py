from collections import defaultdict
from typing import Callable, Dict, List, Set

import torch
import torch.nn as nn
from torch import Tensor
from diffusers.models import UNet2DConditionModel
from diffusers.models.attention_processor import Attention


def find_attn_modules(module: nn.Module):
    """
    Find all attention modules in a module
    """

    return [
        (name, mod)
        for name, mod in module.named_modules()
        if isinstance(module, Attention)
    ]


def get_module_path(parent_module: nn.Module, module: nn.Module) -> str:
    """
    Find the path of a module in a parent module
    :param parent_module: parent module
    :param module: module to find
    :return: path of module in parent_module
    """

    for name, named_module in parent_module.named_modules():
        if named_module == module:
            return name

    return None


def get_module_from_path(parent_module: nn.Module, path: str) -> nn.Module:
    """
    Get a module from a path in a parent module
    :param parent_module: parent module
    :param path: path to module
    :return: module at path
    """

    cur_module = parent_module
    for component in path.split("."):

        if component.isdigit():
            cur_module = cur_module[int(component)]
        else:
            cur_module = getattr(cur_module, component)

    return cur_module


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
            del self._named_handles[name]

    def clear_all_hooks(self):
        for handle in self._named_handles.values():
            handle.remove()
        self._named_handles = dict()


class DiffusionFeatureExtractor:

    # store saved features here, list because one per timestep
    _saved_features: Dict[str, List[torch.Tensor]]

    # store "local vars for each hook"
    _hook_data: Dict[str, dict]

    save_steps: list

    def __init__(self, save_steps=None) -> None:
        self.hook_manager = HookManager()
        self._saved_features = defaultdict(lambda: [])
        self._hook_data = dict()

        if save_steps is None:
            save_steps = []
        self.save_steps = save_steps

    def clear_features(self):
        self._saved_features = defaultdict(lambda: [])

    def add_save_feature_hook(self, name: str, module: nn.Module):
        self.hook_manager.add_named_hook(name, module, self._save_feature_hook(name))

    def get_feature(self, name: str, timestep=0):
        timestep_index = self.save_steps.index(timestep)
        return self._saved_features[name][timestep_index]

    def _save_feature_hook(self, name: str):
        """
        Create a hook that saves the output of a module with key `name`
        """

        self._hook_data[name] = {"cur_step": 0}

        # pylint: disable=unused-argument
        def hook(module, inp, out):

            # save feature if current step is in save_steps
            if self._hook_data[name]["cur_step"] in self.save_steps:
                self._saved_features[name].append(out.cpu().numpy())

            # increment step
            self._hook_data[name]["cur_step"] += 1

        return hook


class SAFeatureExtractor:

    def __init__(self) -> None:
        self.hooks = HookManager()
        self.saved_outputs = dict()
        self.saved_inputs = dict()

    def _post_attn_hook(self, module_name: str):
        # pylint: disable=unused-argument
        def hook(module, inp, output):
            # self.saved_inputs[module_name] = inp[0].cpu()
            self.saved_outputs[module_name] = output.cpu()

        return hook

    def _pre_attn_hook(self, module_name: str):
        # pylint: disable=unused-argument
        def hook(module, inp, output):
            self.saved_inputs[module_name] = inp[0].cpu()

        return hook

    def add_attn_hooks(self, attn: Attention, name: str):
        self.hooks.add_named_hook(f"save_{name}_out", attn, self._post_attn_hook(name))
        self.hooks.add_named_hook(
            f"save_{name}_in", attn.to_k, self._pre_attn_hook(name)
        )
