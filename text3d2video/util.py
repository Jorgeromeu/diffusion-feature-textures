from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from pytorch3d.io import load_obj, load_objs_as_meshes
from torch import Tensor
from tqdm import tqdm

from text3d2video.utilities.mesh_processing import normalize_meshes


def read_obj_uvs(obj_path: str, device="cuda"):
    _, faces, aux = load_obj(obj_path)
    verts_uvs = aux.verts_uvs.to(device)
    faces_uvs = faces.textures_idx.to(device)
    return verts_uvs, faces_uvs


def read_obj_with_uvs(obj_path: str, device="cuda", normalize=True):
    _, faces, aux = load_obj(obj_path)
    verts_uvs = aux.verts_uvs.to(device)
    faces_uvs = faces.textures_idx.to(device)

    mesh = load_objs_as_meshes([obj_path], device=device)
    mesh = normalize_meshes(mesh)
    return mesh, verts_uvs, faces_uvs


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
    return np.array(indices, dtype=int)


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


def unique_with_indices(tensor: Tensor, dim: int = 0) -> Tuple[Tensor, Tensor]:
    unique, inverse = torch.unique(tensor, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    unique_indices = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return unique, unique_indices


def dict_map(dict: Dict, callable: Callable) -> Dict:
    return {k: callable(k, v) for k, v in dict.items()}


def object_array(list_of_lists: List):
    # determine how many dimensions in LoL
    n_dims = 0
    lst = list_of_lists
    while isinstance(lst, list):
        lst = lst[0]
        n_dims += 1

    # determine shape of LoL
    shape = []
    lst = list_of_lists
    for i in range(n_dims):
        dim_len = len(lst)
        shape.append(dim_len)
        lst = lst[0]

    # create empty
    arr = np.empty(shape, dtype=object)

    # populate
    for idxs, _ in np.ndenumerate(arr):
        value = list_of_lists
        for i in idxs:
            value = value[i]

        arr[idxs] = value

    return arr


def pil_latent(latent: Tensor):
    return TF.to_pil_image(latent[0:3].cpu())


def chunk_dim(x: Tensor, n_chunks: int, dim: int = 0):
    """
    Chunk a tensor along a dimension
    """

    size = x.size(dim)
    assert size % n_chunks == 0, "dim length size must be divisible by n_chunks"
    chunk_size = size // n_chunks
    new_shape = list(x.shape[:dim]) + [n_chunks, chunk_size] + list(x.shape[dim + 1 :])
    return x.view(*new_shape)


def hwc_to_chw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert HWC (or NHWC) image format to CHW (or NCHW).
    Supports single image (3D tensor) or batched images (4D tensor).
    """
    if x.ndim == 3:  # HWC
        return x.permute(2, 0, 1)
    elif x.ndim == 4:  # NHWC
        return x.permute(0, 3, 1, 2)
    else:
        raise ValueError("Input tensor must be 3D (HWC) or 4D (NHWC)")


def chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    """
    Convert CHW (or NCHW) image format to HWC (or NHWC).
    Supports single image (3D tensor) or batched images (4D tensor).
    """
    if x.ndim == 3:  # CHW
        return x.permute(1, 2, 0)
    elif x.ndim == 4:  # NCHW
        return x.permute(0, 2, 3, 1)
    else:
        raise ValueError("Input tensor must be 3D (CHW) or 4D (NCHW)")


def map_array(arr: np.ndarray, map_func: Callable, pbar=False) -> np.ndarray:
    if pbar:
        arr_flat = arr.flatten()
        B_flat = np.array([map_func(x) for x in tqdm(arr_flat)])
        B = B_flat.reshape(arr.shape)
        return B

    return np.vectorize(map_func)(arr)


def group_into_array(entries: List, key_funs: List[Callable]) -> Dict:
    # for each key_fun get set of values
    dim_values = [set() for _ in key_funs]
    for r in entries:
        for i, key_fun in enumerate(key_funs):
            key = key_fun(r)
            dim_values[i].add(key)

    dim_values = [sorted(list(d)) for d in dim_values]
    shape = [len(d) for d in dim_values]

    # create empty grid
    grid = np.empty(shape, dtype=object)

    for e in entries:
        # get keys
        keys = tuple(key_fun(e) for key_fun in key_funs)

        indices = []
        for i, key in enumerate(keys):
            idx = dim_values[i].index(key)
            indices.append(idx)

        index = tuple(indices)

        # populate grid
        grid[index] = e

    return grid, dim_values


def split_into_chunks(x, chunk_size=5):
    if isinstance(x, torch.Tensor):
        return x.split(chunk_size)
    elif isinstance(x, list):
        return [x[i : i + chunk_size] for i in range(0, len(x), chunk_size)]
    else:
        raise TypeError(f"Unsupported type: {type(x)}")
