from dataclasses import dataclass
from math import sqrt
from typing import List

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import einsum, rearrange
from jaxtyping import Float
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from PIL.Image import Image
from torch import Tensor

from text3d2video.feature_visualization import RgbPcaUtil


@dataclass
class AttnFeatures:
    qrys_mh: Tensor
    keys_mh: Tensor
    vals_mh: Tensor
    layer_res: int


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


def concatenate_images(images: List[Image]):
    val_images = torch.stack([TF.to_tensor(img) for img in images])
    val_image = rearrange(val_images, "b c h w -> c h (b w)")
    return TF.to_pil_image(val_image)


def coord_to_pixel(coord, size):
    """
    Convert cartesian coordinate to pixel coordinates
    :param coord: (2,) tensor of normalized coordinates [0,1]
    :param size: (2,) tensor of image size
    """

    coord_pix = coord.clone()
    coord_pix[0] = coord_pix[0] * size[0]
    coord_pix[1] = coord_pix[1] * size[1]
    coord_pix = coord_pix.long()

    return coord_pix


def reshape_concatenated(
    seq: Tensor,
    layer_res: int,
):
    """
    Reshape a key/value to a square grid
    :param seq: (n, T) tensor
    :param layer_res: resolution of the layer before flattening/concatenating
    : return: (n, h, w) tensor
    """
    return rearrange(seq, "(n h w) d -> d h (n w)", w=layer_res, h=layer_res)


def pixel_coord_flattened(pix_coord: Tensor, size: Tensor):
    """
    Convert pixel coordinates to flattened index
    :param pix_coord: (2,) tensor of pixel coordinates
    :param size: (2,) tensor of image size
    """
    return pix_coord[0] + pix_coord[1] * size[0]


def plot_image_and_weight(
    ax: Axes,
    img: Image,
    weights: Tensor,
    alpha=0.8,
    interpolation="bicubic",
    cmap="turbo",
):
    """
    Plot an image with a heatmap overlay
    :param ax: matplotlib axes
    :param img: PIL image
    :param weights: (h, w) tensor of weights
    """

    if img is not None:
        im_imshow = ax.imshow(img)
        w, h = img.size
    else:
        im_imshow = None
        w, h = weights.shape

    if weights is not None:
        weights_imshow = ax.imshow(
            weights,
            cmap=cmap,
            alpha=alpha,
            interpolation=interpolation,
            extent=[0, w, h, 0],
        )
    return im_imshow, weights_imshow


def add_pixel_marker(
    ax: Axes, coord: Tensor, bg_img_size, grid_img_size, color="lime", opacity=0.8
):
    """
    Add a pixel marker to a maptlotlib axes
    :param ax: matplotlib axes
    :param coord: (2,) tensor of image coordinates [0,1] top-left
    :param bg_img_size: (2,) shape of background image
    : param layer_res: resolution of the layer
    """

    # get coordinate in layer_res pixel space
    coord_pix = coord_to_pixel(coord, grid_img_size)

    # convert to image space
    pixel_w = bg_img_size[0] / grid_img_size[0]
    coord_pix = coord_pix * pixel_w

    # create rect
    rect = Rectangle(coord_pix, height=pixel_w, width=pixel_w, color=color)
    return ax.add_patch(rect)


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
            attn_scores / (temperature * sqrt(qrys.shape[-1])), dim=1
        )
        return attn_weights.cpu()


def plot_qry_weights(
    ax_qry: Axes,
    ax_kv: Axes,
    attn_features: AttnFeatures,
    attn_weights: Tensor,
    qry_coord: Tensor,
    qry_frame_idx: int,
    target_frame_indices: List[int],
    images: List[Image],
    weight_only=False,
):
    """
    Plot attention attention weights and query position over generated images
    :param ax_qry: matplotlib axes for query image
    :param ax_kv: matplotlib axes for key/value images
    :param attn_features: AttnFeatures object
    :param attn_weights: (b, t_q, t_kv) tensor of attention weights
    :param qry_coord: (2,) tensor of query coordinates
    :param qry_frame_idx: index of query frame
    :param target_frame_indices: indices of key/value frames
    :param images: list of PIL images
    """

    qry_grid_size = (attn_features.layer_res, attn_features.layer_res)

    if ax_qry is not None:
        qry_im = images[qry_frame_idx]
        qry_imshow = ax_qry.imshow(qry_im)
        pixel_marker = add_pixel_marker(ax_qry, qry_coord, qry_im.size, qry_grid_size)

    if ax_kv is not None:
        # get index in attn_weights for query pixel
        qry_pix = coord_to_pixel(qry_coord, qry_grid_size)
        qry_pix_1d = pixel_coord_flattened(qry_pix, qry_grid_size)

        # get weights for quer and reshape to square
        weights = attn_weights[qry_frame_idx, qry_pix_1d, :]
        weights = reshape_concatenated(
            weights.unsqueeze(-1), layer_res=attn_features.layer_res
        )[0]

        kv_im = concatenate_images([images[idx] for idx in target_frame_indices])

        img_input = None if weight_only else kv_im
        im_imshow, weight_imshow = plot_image_and_weight(ax_kv, img_input, weights)

    return qry_imshow, pixel_marker, im_imshow, weight_imshow


def make_gridspec_figure(
    n_rows: int,
    n_cols: int,
    width_ratios: List[int],
    height_ratios: List[int],
    scale: int = 2,
):
    height = sum(height_ratios) + 0.2
    width = sum(width_ratios)

    fig = plt.figure(figsize=(width * scale, height * scale))
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=0.0,
        wspace=0.01,
    )

    return fig, gs


def plot_attention_weights(
    ax_qry: Axes,
    ax_kv: Axes,
    qry: Tensor,
    key: Tensor,
    layer_res: int,
    qry_coord: Tensor,
    qry_im,
    kv_im=None,
):
    qry_grid_size = (layer_res, layer_res)

    # plot query image and marker
    ax_qry.imshow(qry_im)
    add_pixel_marker(ax_qry, qry_coord, qry_im.size, qry_grid_size)

    # get query pixel coordinate
    qry_pix = coord_to_pixel(qry_coord, qry_grid_size)
    qry_pix_1d = pixel_coord_flattened(qry_pix, qry_grid_size)

    # compute attn weights
    attn_weights = compute_attn_weights(qry.unsqueeze(0), key.unsqueeze(0))[0]
    weights = attn_weights[qry_pix_1d, :]
    weights = reshape_concatenated(weights.unsqueeze(-1), layer_res=layer_res)[0]

    # plot weights
    plot_image_and_weight(ax_kv, kv_im, weights, interpolation="nearest", alpha=0.8)


def plot_qry_and_coord(ax_qry: Axes, qry_im: Image, qry_coord: Tensor, layer_res: int):
    qry_grid_size = (layer_res, layer_res)
    ax_qry.imshow(qry_im)
    add_pixel_marker(ax_qry, qry_coord, qry_im.size, qry_grid_size)
