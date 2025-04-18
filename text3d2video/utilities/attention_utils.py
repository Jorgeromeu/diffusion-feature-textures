from math import sqrt
from typing import List

import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


def memory_efficient_attention(
    attn: Attention, key, query, value, attention_mask, temperature=1.0
):
    """
    Attention operation with F.scaled_dot_product_attention
    """

    batch_size = query.shape[0]

    inner_dim = key.shape[-1]
    head_dim = inner_dim // attn.heads

    query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
    value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

    if attn.norm_q is not None:
        query = attn.norm_q(query)
    if attn.norm_k is not None:
        key = attn.norm_k(key)

    d_kv = key.shape[-1]

    scale = 1 / (temperature * sqrt(d_kv))

    # pylint: disable=not-callable
    hidden_states = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )

    hidden_states = hidden_states.transpose(1, 2).reshape(
        batch_size, -1, attn.heads * head_dim
    )
    attn_out = hidden_states.to(query.dtype)

    return attn_out


def extended_attn_kv_hidden_states(
    x: Float[Tensor, "b t d"], chunk_size: int = 2, frame_indices: List[int] = None
) -> Float[Tensor, "b t d"]:
    n_frames = x.shape[0] // chunk_size

    # unstack batch-frame dimension
    extended_x = rearrange(x, "(b f) t c -> b f t c", f=n_frames)

    # select specific frames
    if frame_indices is not None:
        extended_x = extended_x[:, frame_indices, ...]

    # stack frames across seq dimension
    extended_x = rearrange(extended_x, "b f t c -> b (f t) c")

    return extended_x


def extend_across_frame_dim(
    x: Float[Tensor, "b t d"], n_frames: int
) -> Float[Tensor, "b* t d"]:
    """
    Given a tensor B T D for cross-frame attention, expand across
    frame dimension, and stack as batch dimension
    :param x: (batch_size, sequence_length, hidden_size)
    return: (batch_size * n_frames, sequence_length, hidden_size)
    """

    expanded = x.unsqueeze(1).expand(-1, n_frames, -1, -1)
    return rearrange(expanded, "b f t d -> (b f) t d")


def reshape_kvs_2D(x: Float[Tensor, "b t d"], res: int) -> Float[Tensor, "b t d"]:
    """
    Given KV features, and the resolution of original sequence reshape to 2D grid
    """

    # return rearrange(x, "b (h w) c -> b c h w", w=res)
    return rearrange(x, "b (n h w) c -> b c h (n w)", w=res, h=res)


def expand_singleton_dim(x: Tensor, n: int, dim=0) -> Tensor:
    shape = list(x.shape)
    assert shape[dim] == 1, f"Dimension {dim} is not singleton"
    shape[dim] = n
    return x.expand(*shape)
