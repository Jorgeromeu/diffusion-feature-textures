from typing import List, Optional

import torch
from diffusers.models.attention_processor import Attention
from torch import Tensor

from text3d2video.sd_feature_extraction import get_module_path
from text3d2video.utilities.attention_utils import (
    memory_efficient_attention,
)


class BaseAttnProcessor:
    """
    Base attention processor that we can easily override with custom behavior
    """

    # internal state
    _cur_module_path: str
    _is_cross_attn: bool
    _is_self_attn: bool

    # optionally hold information about what is being denoised:
    cur_timestep: int = None
    frame_indices: Tensor = None

    # treat batch dimension as chunks, (e.g for CFG)
    chunk_labels: List[str] = []

    def set_cur_timestep(self, cur_timestep: int):
        self.cur_timestep = cur_timestep

    def set_frame_indices(self, frame_indices: Tensor):
        self.frame_indices = frame_indices

    def set_chunk_labels(self, chunk_labels: List[str]):
        self.chunk_labels = chunk_labels

    def n_chunks(self) -> int:
        """
        Return the number of chunks
        """
        return len(self.chunk_labels)

    def __init__(
        self,
        model,
    ):
        self.model = model

    def _call_init(self, attn: Attention, encoder_hidden_states: Tensor):
        self._cur_module_path = get_module_path(self.model, attn)
        self._is_cross_attn = encoder_hidden_states is not None
        self._is_self_attn = not self._is_cross_attn

    def _call_cross_attn(
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
        val = attn.to_v(kv_hidden_states)

        return memory_efficient_attention(attn, key, qry, val, attention_mask)

    def _call_self_attn(
        self, attn: Attention, hidden_states: Tensor, attention_mask: Tensor
    ):
        key = attn.to_k(hidden_states)
        val = attn.to_v(hidden_states)
        qry = attn.to_q(hidden_states)

        return memory_efficient_attention(attn, key, qry, val, attention_mask)

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

        self._call_init(attn, encoder_hidden_states)

        if self._is_cross_attn:
            attn_out = self._call_cross_attn(
                attn, hidden_states, encoder_hidden_states, attention_mask
            )
        else:
            attn_out = self._call_self_attn(attn, hidden_states, attention_mask)

        # linear proj to output dim
        attn_out = attn.to_out[0](attn_out)
        attn_out = attn.to_out[1](attn_out)

        if attn.residual_connection:
            attn_out = attn_out + hidden_states

        return attn_out
