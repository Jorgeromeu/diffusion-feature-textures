from math import sqrt
from typing import List

import torch
import torch.nn.functional as F
from einops import einsum, rearrange
from jaxtyping import Float
from matplotlib.axes import Axes
from rerun import Image
from torch import FloatTensor, Tensor

from text3d2video.utilities.attention_visualization import (
    add_pixel_marker,
    coord_to_pixel,
    pixel_coord_flattened,
    reshape_concatenated,
)


def split_into_heads(
    seq: Float[Tensor, "b t d"], n_heads=8
) -> Float[Tensor, "b t n_heads head_dim"]:
    """
    Split the last dimension of a sequence into multiple heads.
    """

    return rearrange(
        seq, "b t (n_heads head_dim) -> b t n_heads head_dim", n_heads=n_heads
    )


def compute_attn_weights(
    qrys: Float[Tensor, "b t d"],
    keys: Float[Tensor, "b t d"],
    temperature=1,
    device="cuda",
) -> Float[Tensor, "b t_q t_kv"]:
    """
    Compute attention weights for key/qry pair
    :param qrys: (b, t, d) tensor of queries
    :param keys: (b, t, d) tensor of keys
    :param temperature: temperature for softmax
    :param device: device to compute on
    : return: (b, t_q, t_kv) tensor of attention weights
    """

    with torch.no_grad():
        attn_scores = einsum(
            qrys.to(device), keys.to(device), "b tq d, b tk d -> b tq tk"
        )
        attn_weights = F.softmax(
            attn_scores / (temperature * sqrt(qrys.shape[-1])), dim=2
        )
        return attn_weights.cpu()


def calc_attn_weights_per_head(qry, key, head_idx=0):
    qrys_mh = split_into_heads(qry.unsqueeze(0))
    keys_mh = split_into_heads(key.unsqueeze(0))

    # index out head
    qrys_head = qrys_mh[:, :, head_idx, :]
    keys_head = keys_mh[:, :, head_idx, :]

    # sdp attention
    weights = compute_attn_weights(qrys_head, keys_head)[0]
    return weights


def calc_attn_weights_all_heads(qry, key):
    weights_head = [calc_attn_weights_per_head(qry, key, head_idx=i) for i in range(8)]
    weights_head = torch.stack(weights_head)
    weights_avg = weights_head.mean(dim=0)

    return weights_avg


def plot_sa_weights(
    ax_qry: Axes,
    ax_kv: Axes,
    attn_weights: Float[Tensor, "T_qry T_kv"],
    qry_coord: Tensor,
    qry_img: Image = None,
    kv_img: Image = None,
    pixel_marker=True,
    vmin=None,
    vmax=None,
    marker_color="red",
    alpha=0.95,
):
    res = int(sqrt(attn_weights.shape[0]))

    # get pixel coord corresponding to float-coord
    coord_pix = coord_to_pixel(qry_coord, (res, res))
    coord_pix_flat = pixel_coord_flattened(coord_pix, (res, res))

    # index row in attn_weights
    row = attn_weights[coord_pix_flat, :]

    # reshape to 2D
    attn_weights_2D = reshape_concatenated(row.unsqueeze(-1), res)[0]

    # plot qry image and marker
    if ax_qry is not None:
        if qry_img is not None:
            ax_qry.imshow(qry_img, extent=[0, 1, 1, 0])

        if pixel_marker:
            add_pixel_marker(ax_qry, qry_coord, (1, 1), (res, res), color=marker_color)
        else:
            ax_qry.scatter(
                qry_coord[0],
                qry_coord[1],
                color=marker_color,
                marker=".",
                s=100,
            )

    # plot kv image and attention weights
    if ax_kv is not None:
        if kv_img is not None:
            height, width = attn_weights_2D.shape
            ax_kv.imshow(kv_img, extent=[0, width, height, 0])

        ax_kv.imshow(
            attn_weights_2D,
            cmap="turbo",
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
        )


def plot_ca_weights(
    ax_qry: Axes,
    ax_kv: Axes,
    attn_weights: Float[Tensor, "T_qry T_kv"],
    qry_coord: Tensor,
    qry_img: Image,
    token_indices: Tensor,
    token_labels: List[str],
    pixel_marker=True,
    marker_color="red",
    vmax=None,
):
    res = int(sqrt(attn_weights.shape[0]))

    # get pixel coord corresponding to float-coord
    coord_pix = coord_to_pixel(qry_coord, (res, res))
    coord_pix_flat = pixel_coord_flattened(coord_pix, (res, res))

    # index row in attn_weights
    row = attn_weights[coord_pix_flat, :]
    row_cropped = row[token_indices]

    # plot qry image and marker
    if ax_qry is not None:
        ax_qry.imshow(qry_img, extent=[0, 1, 1, 0])

        if pixel_marker:
            add_pixel_marker(ax_qry, qry_coord, (1, 1), (res, res), color=marker_color)
        else:
            ax_qry.scatter(
                qry_coord[0],
                qry_coord[1],
                color=marker_color,
                marker=".",
                s=100,
            )

    # plot bar chart of token weights

    if ax_kv is not None:
        ax_kv.set_yticks([])
        xs = torch.arange(len(token_labels))
        ax_kv.bar(xs, row_cropped, color=marker_color)
        ax_kv.set_xticks(xs)
        ax_kv.set_xticklabels(token_labels, rotation=90)
        if vmax is not None:
            ax_kv.set_ylim(0, vmax)

    # # plot kv image and attention weights
    # if ax_kv is not None:
    #     ax_kv.imshow(kv_img, extent=[0, 1, 0, 1])

    #     ax_kv.imshow(
    #         attn_weights_2D,
    #         extent=[0, 1, 0, 1],
    #         cmap="turbo",
    #         alpha=0.95,
    #         vmin=vmin,
    #         vmax=vmax,
    #     )


def plot_ca_map(ax: Axes, attn_weights: Tensor, token_idx: int, img: Image):
    pass
