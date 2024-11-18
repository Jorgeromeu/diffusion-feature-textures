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
    Convert normalized coordinates to pixel coordinates
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
    interpolation="bilinear",
    cmap="turbo",
):
    """
    Plot an image with a heatmap overlay
    :param ax: matplotlib axes
    :param img: PIL image
    :param weights: (h, w) tensor of weights
    """

    ax.imshow(img)
    if weights is not None:
        w, h = img.size
        ax.imshow(
            weights,
            cmap=cmap,
            alpha=alpha,
            interpolation=interpolation,
            extent=[0, w, h, 0],
        )


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
    ax.add_patch(rect)


def plot_qry_value_weights(
    qry_img,
    val_img,
    qry_weights: Tensor = None,
    kv_weights: Tensor = None,
    qry_coord: Tensor = None,
    val_coord: Tensor = None,
    circle_color="lime",
    weights_cmap="turbo",
    weights_alpha=0.5,
    weights_interpolation="bilinear",
    hide_axes=False,
    layer_res=64,
):
    fig, axs = plt.subplots(1, 2)
    ax_qry = axs[0]
    ax_val = axs[1]
    ax_qry.set_title("Query")
    ax_val.set_title("Key/Value")

    # plot images and weights
    weights_kwargs = {
        "alpha": weights_alpha,
        "interpolation": weights_interpolation,
        "cmap": weights_cmap,
    }
    plot_image_and_weight(ax_qry, qry_img, qry_weights, **weights_kwargs)
    plot_image_and_weight(ax_val, val_img, kv_weights, **weights_kwargs)

    # add markers
    marker_kwargs = {
        "color": circle_color,
    }
    if qry_coord is not None:
        grid_size = (layer_res, layer_res)
        add_pixel_marker(ax_qry, qry_coord, qry_img.size, grid_size, **marker_kwargs)
    if val_coord is not None:
        grid_size = (layer_res * 5, layer_res)
        add_pixel_marker(ax_val, val_coord, val_img.size, grid_size, **marker_kwargs)

    # hide axes
    if hide_axes:
        ax_qry.axis("off")
        ax_val.axis("off")

    # set size
    scale = 0.5
    ax_qry.set_position([0, 0, scale, scale])
    ax_val.set_position([scale, 0, scale * 3.8, scale])
