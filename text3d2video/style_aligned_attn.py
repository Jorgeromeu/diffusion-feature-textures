from enum import Enum
from math import sqrt
from typing import Dict, Optional

import torch
from diffusers.models.attention_processor import Attention
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from text3d2video.artifacts.gr_data import GrDataArtifact
from text3d2video.attention_utils import (
    extend_across_frame_dim,
    extended_attn_kv_hidden_states,
    memory_efficient_attention,
)
from text3d2video.generative_rendering.configs import (
    GenerativeRenderingConfig,
)
from text3d2video.sd_feature_extraction import get_module_path
from text3d2video.util import blend_features


class GrAttnMode(Enum):
    FEATURE_EXTRACTION: str = "extraction"
    FEATURE_INJECTION: str = "injection"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False


class StyleAlignedAttentionProcessor:
    _is_cross_attn: bool
    _is_self_attn: bool

    def __init__(
        self,
        unet_chunk_size=2,
    ):
        """
        :param unet_chunk_size:
            number of batches for each generated image, 2 for classifier free guidance
        """
        self.unet_chunk_size = unet_chunk_size

    # functionality

    def call_init(self, attn: Attention, encoder_hidden_states: Tensor):
        self._is_cross_attn = encoder_hidden_states is not None
        self._is_self_attn = not self._is_cross_attn

    def call_cross_attn(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor],
    ):
        qry = attn.to_q(hidden_states)

        kv_hidden_states = encoder_hidden_states
        if attn.norm_cross is not None:
            kv_hidden_states = attn.norm_cross(hidden_states)

        key = attn.to_k(kv_hidden_states)
        value = attn.to_v(kv_hidden_states)

        return memory_efficient_attention(attn, key, qry, value, attention_mask)

    def call_self_attn(
        self, attn: Attention, hidden_states: Tensor, attention_mask: Tensor
    ):
        n_frames = hidden_states.shape[0] // self.unet_chunk_size
        unstacked_x = rearrange(hidden_states, "(b f) t c -> b f t c", f=n_frames)

        x_ref = unstacked_x[:, 2, ...]
        x_ref = extend_across_frame_dim(x_ref, n_frames)

        kv_seq = x_ref

        key = attn.to_k(kv_seq)
        value = attn.to_v(kv_seq)
        qry = attn.to_q(hidden_states)

        return memory_efficient_attention(attn, key, qry, value, attention_mask)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ):
        """
        :param attn: Attention module
        :param hidden_states: input features
        :param encoder_hidden_states: input features for cross-attention
        :param attention_mask: attention mask
        """

        self.call_init(attn, encoder_hidden_states)

        if self._is_cross_attn:
            attn_out = self.call_cross_attn(
                attn, hidden_states, encoder_hidden_states, attention_mask
            )

        else:
            attn_out = self.call_self_attn(attn, hidden_states, attention_mask)

        # linear proj to output dim
        attn_out = attn.to_out[0](attn_out)
        attn_out = attn.to_out[1](attn_out)

        return attn_out
