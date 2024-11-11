from math import sqrt

import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from einops import einsum, rearrange
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch import Tensor

from text3d2video.feature_visualization import reduce_feature_map


def visualize_attention_weights(
    x: Tensor,
    qrys: Tensor,
    keys: Tensor,
    vals: Tensor,
    attn: Attention,
    pixel_coord: Tensor,
    head_idx: int,
    normalize_weights: bool = True,
    temperature=1.0,
):
    n_heads = attn.heads

    layer_res = int(sqrt(qrys.shape[0]))
    inner_dim = keys.shape[-1]
    head_dim = inner_dim // n_heads

    qrys_multihead = qrys.view(-1, n_heads, head_dim).transpose(1, 2)
    keys_multihead = keys.view(-1, n_heads, head_dim).transpose(1, 2)
    vals_multihead = vals.view(-1, n_heads, head_dim).transpose(1, 2)

    # get pixel index in flattened tensor
    pixel_idx_flat = pixel_coord[1] * layer_res + pixel_coord[0]

    # get query embedding for pixel and head
    pixel_qry = qrys_multihead[pixel_idx_flat, :, head_idx]

    # get keys and values for the given head
    head_keys = keys_multihead[:, :, head_idx]
    head_vals = vals_multihead[:, :, head_idx]

    # compute weights
    weights = einsum(pixel_qry, head_keys, "d , n d -> n")
    if normalize_weights:
        weights = F.softmax(weights / (temperature * sqrt(head_dim)))

    # reshape x to rectangle
    x_square = rearrange(x, "(h w) d -> d h w", h=layer_res)

    # reshape weights and values
    weights_square = rearrange(weights, "(h w) -> h w", w=layer_res)
    vals_square = rearrange(head_vals, "(h w) d -> d h w", w=layer_res)

    _, axs = plt.subplots(1, 3, figsize=(10, 10))

    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])

    ax_x = axs[0]
    ax_x.set_title("Input Features")

    ax_weights = axs[1]
    ax_vals = axs[2]

    square = Rectangle(pixel_coord - 0.5, width=0.9, height=0.9, color="blue")
    ax_x.add_patch(square)

    ax_x.imshow(reduce_feature_map(x_square))
    ax_weights.imshow(weights_square)
    ax_weights.set_title("Weights")

    ax_vals.imshow(reduce_feature_map(vals_square))
    ax_vals.set_title("Values")

    plt.tight_layout()
