from typing import Optional

import torch
from diffusers.models.attention_processor import Attention
from einops import rearrange
from torch import Tensor

from text3d2video.sd_feature_extraction import get_module_path
from text3d2video.utilities.attention_utils import (
    memory_efficient_attention,
)
from text3d2video.utilities.diffusion_data import AttnFeaturesWriter


class DefaultAttnProcessor:
    """
    Base attention processor that we can easily override with custom behavior
    """

    # internal state
    _cur_module_path: str
    _is_cross_attn: bool
    _is_self_attn: bool

    # optionally hold the current timestep
    cur_timestep: int = None

    # optionally hold the attention writer and chunk frame indices for saving qkv
    attn_writer: AttnFeaturesWriter = None
    chunk_frame_indices: Tensor = None

    # public

    def set_attn_data_writer(self, attn_writer: AttnFeaturesWriter):
        self.attn_writer = attn_writer

    def set_cur_timestep(self, cur_timestep: int):
        self.cur_timestep = cur_timestep

    def set_chunk_frame_indices(self, chunk_frame_indices: Tensor):
        self.chunk_frame_indices = chunk_frame_indices

    # protected

    def write_qkv(self, qry: Tensor, key: Tensor, val: Tensor):
        if self.attn_writer is None:
            return

        unstacked_q = rearrange(qry, "(b f) t c -> b f t c", b=self.chunk_size)
        unstacked_k = rearrange(key, "(b f) t c -> b f t c", b=self.chunk_size)
        unstacked_v = rearrange(val, "(b f) t c -> b f t c", b=self.chunk_size)

        timestep = self.cur_timestep
        chunk_frame_indices = self.chunk_frame_indices

        if timestep is None or chunk_frame_indices is None:
            raise ValueError(
                "Timestep and chunk_frame_indices must be set to write qkv"
            )

        self.attn_writer.write_qkv_batched(
            timestep,
            self._cur_module_path,
            unstacked_q,
            unstacked_k,
            unstacked_v,
            chunk_frame_indices=chunk_frame_indices,
        )

    def write_y(self, y: Tensor):
        if self.attn_writer is None:
            return

        unstacked_y = rearrange(y, "(b f) t c -> b f t c", b=self.chunk_size)

        timestep = self.cur_timestep
        chunk_frame_indices = self.chunk_frame_indices

        if timestep is None or chunk_frame_indices is None:
            raise ValueError(
                "Timestep and chunk_frame_indices must be set to write qkv"
            )

        self.attn_writer.write_attn_out_batched(
            timestep,
            self._cur_module_path,
            unstacked_y,
            chunk_frame_indices=chunk_frame_indices,
        )

    def __init__(
        self,
        model,
        chunk_size=2,
    ):
        """
        :param unet_chunk_size:
            number of batches for each generated image, 2 for classifier free guidance
        """
        self.model = model
        self.chunk_size = chunk_size

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

        self.write_qkv(qry, key, val)
        y = memory_efficient_attention(attn, key, qry, val, attention_mask)
        self.write_y(y)
        return y

    def _call_self_attn(
        self, attn: Attention, hidden_states: Tensor, attention_mask: Tensor
    ):
        key = attn.to_k(hidden_states)
        val = attn.to_v(hidden_states)
        qry = attn.to_q(hidden_states)

        self.write_qkv(qry, key, val)
        y = memory_efficient_attention(attn, key, qry, val, attention_mask)
        self.write_y(y)
        return y

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
