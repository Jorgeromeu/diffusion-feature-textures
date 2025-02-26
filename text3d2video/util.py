from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor


def ordered_sample_indices(lst, n):
    """
    Sample n elements from a list in order, return indices
    """

    if n == 0:
        return []
    if n == 1:
        return [0]
    if n >= len(lst):
        return list(range(len(lst)))
    # Calculate the step size based on list length and number of samples
    step = (len(lst) - 1) / (n - 1)
    # Use the calculated step to select indices
    indices = [round(i * step) for i in range(n)]
    return indices


def ordered_sample(lst, n):
    """
    Sample n elements from a list in order.
    """
    indices = ordered_sample_indices(lst, n)
    return [lst[i] for i in indices]


def sample_feature_map_ndc(feature_map: Tensor, coords: Tensor, mode="nearest"):
    """
    Sample the feature map at the given coordinates
    :param feature_map: (C, H, W) feature map
    :param coords: (N, 2) coordinates in the range [-1, 1] (NDC)
    :param mode: interpolation mode
    :return: (N, C) sampled features
    """
    coords = coords.clone()
    coords[:, 1] *= -1
    batched_feature_map = rearrange(feature_map, "c h w -> 1 c h w").to(torch.float32)
    grid = rearrange(coords, "n d -> 1 1 n d")
    out = F.grid_sample(batched_feature_map, grid, align_corners=True, mode=mode)
    out_features = rearrange(out, "1 f 1 n -> n f")
    return out_features.to(feature_map)


def blend_features(
    features_original: Tensor,
    features_rendered: Tensor,
    alpha: float,
    channel_dim=0,
):
    # compute mask, where features_rendered is not zero
    masks = torch.sum(features_rendered, dim=channel_dim, keepdim=True) != 0

    # blend features
    blended = alpha * features_rendered + (1 - alpha) * features_original

    original_background = features_original * ~masks
    blended_masked = blended * masks

    # return blended features, where features_rendered is not zero
    return original_background + blended_masked


def assert_valid_tensor_shape(shape: Tuple):
    for expected_len in shape:
        assert isinstance(
            expected_len, (int, type(None), str)
        ), f"Dimension length must be int, None or str, received {expected_len}"


def assert_tensor_shape(
    t: Tensor, shape: tuple[int, ...], named_dim_sizes: dict[str, int] = None
):
    error_str = f"Expected tensor of shape {shape}, got {t.shape}"

    assert_valid_tensor_shape(shape)
    assert t.ndim == len(shape), f"{error_str}, wrong number of dimensions"

    if named_dim_sizes is None:
        named_dim_sizes = {}

    for dim_i, expected_len in enumerate(shape):
        true_len = t.shape[dim_i]

        # any len is allowed for None
        if expected_len is None:
            continue

        # assert same length as other dims with same key
        if isinstance(expected_len, str):
            # if symbol length not saved, save it
            if expected_len not in named_dim_sizes:
                named_dim_sizes[expected_len] = true_len
                continue

            expected_named_dim_size = named_dim_sizes[expected_len]
            assert (
                named_dim_sizes[expected_len] == true_len
            ), f"{error_str}, expected {expected_named_dim_size} for dimension {expected_len}, got {true_len}"

    return named_dim_sizes


def assert_tensor_shapes(tensors, named_dim_sizes: Dict[str, int] = None):
    if named_dim_sizes is None:
        named_dim_sizes = {}

    for tensor, shape in tensors:
        named_dim_sizes = assert_tensor_shape(tensor, shape, named_dim_sizes)
