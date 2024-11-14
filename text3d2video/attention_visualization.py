from math import sqrt

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Rectangle
from torch import Tensor

from text3d2video.feature_visualization import RgbPcaUtil


def split_heads(seq, n_heads=8):
    """
    Split the last dimension of a sequence into multiple heads.
    """

    return rearrange(
        seq, "b t (n_heads head_dim) -> b t n_heads head_dim", n_heads=n_heads
    )


def compute_sdp_attn_weights(
    query: Float[Tensor, "d"],  # noqa: F821
    keys: Float[Tensor, "t d"],
    temperature: float = 1.0,
):
    weights = einsum(query, keys, "d , n d -> n")
    weights = F.softmax(weights / (temperature * sqrt(query.shape[-1])), dim=0)
    return weights


def pca_qkv(q_map, k_map, v_map):
    """
    Compute PCA on Q/K/V maps
    """

    # fit PCA on Q/K
    q_features = rearrange(q_map, "d h w -> (h w) d")
    k_features = rearrange(k_map, "d h w -> (h w) d")
    qk_features = torch.cat([q_features, k_features])
    pca_qk = RgbPcaUtil.init_from_features(qk_features)

    # fit PCA on V
    v_features = rearrange(v_map, "d h w -> (h w) d")
    pca_v = RgbPcaUtil.init_from_features(v_features)

    # transform Q/K/V
    q_rgb = pca_qk.feature_map_to_rgb(q_map)
    k_rgb = pca_qk.feature_map_to_rgb(k_map)
    v_rgb = pca_v.feature_map_to_rgb(v_map)

    return q_rgb, k_rgb, v_rgb


def reshape_kv_weights(keys, vals, weights, reshape_fun):
    weights_square = reshape_fun(weights.unsqueeze(1))[0]
    k_square = reshape_fun(keys)
    v_square = reshape_fun(vals)
    return k_square, v_square, weights_square


def plot_mpl_qkv_weights(
    qry_square: Tensor,
    key_square: Tensor,
    val_square: Tensor,
    weights_square: Tensor,
    pixel_coord: Tensor,
):
    # apply PCA on Q/K/V feature maps
    q_rgb, k_rgb, v_rgb = pca_qkv(qry_square, key_square, val_square)

    scale = 3
    fig, axs = plt.subplots(1, 4, figsize=(4 * scale, scale))

    ax_qry = axs[0]
    ax_key = axs[1]
    ax_val = axs[2]
    ax_weights = axs[3]

    for ax in [ax_key, ax_val, ax_weights]:
        ax.set_xticks([])
        ax.set_yticks([])

    # set titles
    ax_key.set_title("Key")
    ax_qry.set_title("Query")
    ax_val.set_title("Value")
    ax_weights.set_title("Weights")

    # plot images
    ax_qry.imshow(q_rgb.permute(1, 2, 0), extent=[-1, 1, 1, -1])
    ax_key.imshow(k_rgb.permute(1, 2, 0))
    ax_val.imshow(v_rgb.permute(1, 2, 0))
    ax_weights.imshow(weights_square, cmap="inferno")

    # add marker on qry axis
    patch = Circle(pixel_coord, 0.03, color="red")
    ax_qry.add_patch(patch)

    return fig
