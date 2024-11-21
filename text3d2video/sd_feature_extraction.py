import re
from typing import Callable, Dict, Set

import torch
from diffusers.models.attention_processor import Attention
from torch import Tensor, nn


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


def parse_layer_name(layer_name):
    """
    Parse a layer name into its components
    :param layer_name: layer name
    :return: dict with block_type, attention_index, block_index, attention_index
    """

    match = re.match(
        r"(?:(down_blocks|up_blocks)\.(\d+)|mid_block)\.attentions\.(\d+)\.transformer_blocks\.(\d+)\.attn1",
        layer_name,
    )

    if not match:
        raise ValueError(f"Invalid layer name format: {layer_name}")

    block_type = match.group(1) if match.group(1) else "mid"
    block_index = int(match.group(2)) if match.group(2) else None
    attention_index = int(match.group(3))

    return {
        "block_type": block_type,
        "block_index": block_index,
        "attention_index": attention_index,
    }


def flatten_layer_index(layer_dict):
    block_type = layer_dict["block_type"]
    block_idx = layer_dict["block_index"]
    attn_idx = layer_dict["attention_index"]

    if block_type == "mid":
        return attn_idx

    if block_type == "down_blocks":
        return (block_idx - 1) * 2 + attn_idx

    if block_type == "up_blocks":
        return (block_idx - 1) * 3 + attn_idx


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
        self._named_handles = {}

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
        self._named_handles = {}
