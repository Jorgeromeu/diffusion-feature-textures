import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

def swap_channel_pos(img_tensor: Tensor):
    if len(img_tensor.shape) == 3:
        return rearrange(img_tensor, 'c h w -> h w c')

    # account for batch being present
    elif len(img_tensor.shape) == 4:
        return rearrange(img_tensor, 'b c h w -> b h w c')

def sample_texture(uvs: Tensor, texture_map: Tensor, mode='bilinear'):
    """
    Sample a texture at certain uv coordinates
    :param uvs: N,2
    :param texture_map: C,H,W
    :param mode: sampling mode
    :return: colors: N,C
    """

    # normalize uvs to -1, 1
    uvs_normalized = uvs * 2 - 1

    # convert to grid
    uvs_grid = uvs_normalized.view(1, -1, 1, 2)

    #  rearrange and flip texture, to match format of grid_sample
    texture_rearranged = rearrange(texture_map, 'c h w -> 1 c h w')
    texture_rearranged = torch.flip(texture_rearranged, dims=[2])

    sample_out = F.grid_sample(
        texture_rearranged,
        uvs_grid,
        align_corners=True,
        mode=mode
    )

    # reshape
    sampled_values = rearrange(sample_out, '1 n c 1 -> c n')

    return sampled_values
