from typing import Optional

import torch
from diffusers.models.attention_processor import Attention
from einops import rearrange
from torch import Tensor

from text3d2video.artifacts.sd_data import SdDataArtifact
from text3d2video.sd_feature_extraction import get_module_path
from text3d2video.utilities.attention_utils import (
    extend_across_frame_dim,
    extended_attn_kv_hidden_states,
    memory_efficient_attention,
)


class StyleAlignedAttentionProcessor:
    cur_timestep = 0
    _cur_module_path: str
    _is_cross_attn: bool
    _is_self_attn: bool
    chunk_frame_indices: Tensor

    data_artifact: SdDataArtifact

    def __init__(
        self,
        unet,
        ref_index: int = 0,
        attend_to: str = "both",
        unet_chunk_size=2,
    ):
        """
        :param unet_chunk_size:
            number of batches for each generated image, 2 for classifier free guidance
        """
        self.unet = unet
        self.ref_index = ref_index
        self.attend_to = attend_to
        self.unet_chunk_size = unet_chunk_size

    # functionality

    def call_init(self, attn: Attention, encoder_hidden_states: Tensor):
        self._cur_module_path = get_module_path(self.unet, attn)
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

        key_self = attn.to_k(hidden_states)
        val_self = attn.to_v(hidden_states)

        x_ref = unstacked_x[:, self.ref_index, ...]
        x_ref = extend_across_frame_dim(x_ref, n_frames)
        key_ref = attn.to_k(x_ref)
        val_ref = attn.to_v(x_ref)
        qry_ref = attn.to_q(x_ref)

        qry = attn.to_q(hidden_states)

        # # adains
        # key_self = adain(key_self, key_ref)
        # val_self = adain(val_self, val_ref)
        # qry = adain(qry, qry_ref)

        if self.attend_to == "self":
            key = key_self
            val = val_self
        if self.attend_to == "reference":
            key = key_ref
            val = val_ref
        if self.attend_to == "both":
            key = torch.cat([key_self, key_ref], dim=1)
            val = torch.cat([val_self, val_ref], dim=1)

        if self.attend_to == "all":
            ext_hidden_states = extended_attn_kv_hidden_states(
                hidden_states, chunk_size=2
            )
            ext_hidden_states = extend_across_frame_dim(ext_hidden_states, n_frames)
            key = attn.to_k(ext_hidden_states)
            val = attn.to_v(ext_hidden_states)

        # save pre attn features
        unstacked_q = rearrange(qry, "(b f) t c -> b f t c", f=n_frames)
        unstacked_k = rearrange(key, "(b f) t c -> b f t c", f=n_frames)
        unstacked_v = rearrange(val, "(b f) t c -> b f t c", f=n_frames)
        self.data_artifact.attn_writer.write_qkv_batched(
            self.cur_timestep,
            self._cur_module_path,
            unstacked_q,
            unstacked_k,
            unstacked_v,
            chunk_frame_indices=self.chunk_frame_indices,
        )

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

        if attn.residual_connection:
            # attn_out = attn_out + hidden_states
            pass

        return attn_out
