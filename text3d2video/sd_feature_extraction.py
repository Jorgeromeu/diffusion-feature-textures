from enum import Enum
from typing import Tuple

from attr import dataclass
from diffusers import UNet2DConditionModel
from diffusers.models.attention_processor import Attention
from torch import nn


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
    def parse(cls, module: str):
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

    def resolution(self, unet: UNet2DConditionModel, input_res=64):
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


def read_layer_paths(
    modules: list[str],
) -> Tuple[list[AttnLayerId], list[AttnLayerId], list[AttnLayerId]]:
    parsed = [AttnLayerId.parse(module) for module in modules]
    parsed = sorted(parsed, key=lambda x: x.unet_absolute_index())
    enc_layers = [layer for layer in parsed if layer.block_type == BlockType.DOWN]
    dec_layers = [layer for layer in parsed if layer.block_type == BlockType.UP]
    mid_layers = [layer for layer in parsed if layer.block_type == BlockType.MID]
    return enc_layers, mid_layers, dec_layers


def find_attn_layers(
    module: nn.Module,
    layer_types: AttnType = None,
    resolutions=None,
    block_types=None,
    return_as_string=True,
):
    if layer_types is None:
        layer_types = [AttnType.SELF_ATTN, AttnType.CROSS_ATTN]

    if resolutions is None:
        resolutions = [64, 32, 16, 8]

    if block_types is None:
        block_types = [BlockType.DOWN, BlockType.MID, BlockType.UP]

    modules = find_attn_modules(module)

    # parse
    layers = [AttnLayerId.parse(m) for m in modules]

    # sort by order
    layers = sorted(layers, key=lambda x: x.unet_absolute_index())

    results = []
    for layer in layers:
        resolution = layer.resolution(module)

        if (
            layer.attn_type in layer_types
            and resolution in resolutions
            and layer.block_type in block_types
        ):
            if return_as_string:
                results.append(layer.module_path())
            else:
                results.append(layer)

    return results
