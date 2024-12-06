from enum import Enum
from typing import Callable, Dict, Set

import torch
from attr import dataclass
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from torch import Tensor, nn


def find_attn_modules(module: nn.Module):
    """
    Find all attention modules in a module
    """

    modules = []

    for name, mod in module.named_modules():
        if isinstance(mod, Attention):
            modules.append(name)

    return modules


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


class AttnType(Enum):
    SELF_ATTN: str = "SA"
    CROSS_ATTN: str = "CA"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False


class BlockType(Enum):
    DOWN: str = "down_blocks"
    UP: str = "up_blocks"
    MID: str = "mid_block"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False


@dataclass
class AttnLayerId:
    block_type: BlockType
    attn_type: AttnType
    block_index: int
    attention_index: int

    @classmethod
    def parse_module_path(cls, module: str):
        components = module.split(".")

        # check if up/down/mid block
        block_type = components[0]
        block_type = BlockType(block_type)

        # indices in components of block_idx and attn_idx
        str_block_idx = 1
        str_attn_idx = 3

        if block_type == BlockType.MID:
            # for mid block, block index is always 0 and attn_index is shiftex
            block_idx = 0
            str_attn_idx -= 1
        else:
            # read block index
            block_idx = int(components[str_block_idx])

        # read attention index
        attention_idx = int(components[str_attn_idx])

        # read attention type (self/cross)
        attn_type_idx = int(components[-1][-1])
        attn_type = AttnType.SELF_ATTN if attn_type_idx == 1 else AttnType.CROSS_ATTN

        return AttnLayerId(
            block_type=block_type,
            block_index=block_idx,
            attention_index=attention_idx,
            attn_type=attn_type,
        )

    def get_attn_layer(self, unet: UNet2DConditionModel):
        return get_module_from_path(unet, self.module_path)

    def module_path(self) -> str:
        block_type_str = self.block_type.value
        attn_type_idx = 1 if self.attn_type == AttnType.SELF_ATTN else 2

        if block_type_str == "mid_block":
            return f"{block_type_str}.attentions.{self.attention_index}.transformer_blocks.0.attn{attn_type_idx}"
        return f"{block_type_str}.{self.block_index}.attentions.{self.attention_index}.transformer_blocks.0.attn{attn_type_idx}"

    def level_idx(self, unet: UNet2DConditionModel):
        block_idx = self.block_index
        n_levels = len(unet.down_blocks)

        if self.block_type == BlockType.DOWN:
            return block_idx

        if self.block_type == BlockType.UP:
            return n_levels - block_idx - 1

        if self.block_type == BlockType.MID:
            return n_levels - 1

    def layer_channels(self, unet: UNet2DConditionModel):
        return unet.block_out_channels[self.level_idx(unet)]

    def layer_resolution(self, unet: UNet2DConditionModel, input_res=64):
        """
        Gives the resolution the attn layer operates at
        """

        level_idx = self.level_idx(unet)
        res = input_res // (2**level_idx)
        return res

    def unet_path_index(self):
        """
        Gives the index of the attention layer, in its path (enc/dec/mid)
        """

        block_type = self.block_type
        block_idx = self.block_index
        attn_idx = self.attention_index

        # TODO make more agnostic to unet architecture

        if block_type == BlockType.DOWN:
            return (block_idx) * 2 + attn_idx

        if block_type == BlockType.UP:
            return (block_idx - 1) * 3 + attn_idx

        if block_type == BlockType.MID:
            return attn_idx

    def unet_absolute_index(self):
        """
        Gives the absolute index of the attention layer in the UNet
        """

        # TODO make more agnostic to unet architecture

        path_idx = self.unet_path_index()

        if self.block_type == BlockType.DOWN:
            return path_idx
        if self.block_type == BlockType.MID:
            return path_idx + 6
        if self.block_type == BlockType.UP:
            return path_idx + 7


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
