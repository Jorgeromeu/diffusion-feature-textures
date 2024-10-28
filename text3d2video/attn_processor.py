from dataclasses import dataclass
from typing import Optional

import torch
from diffusers.models.attention_processor import Attention
from jaxtyping import Float

from text3d2video.attention_utils import (
    averaged_attn_kv_hidden_states,
    extend_across_frame_dim,
    extended_attn_kv_hidden_states,
    memory_efficient_attention,
)
from text3d2video.multidict import MultiDict2
from text3d2video.sd_feature_extraction import get_module_path


@dataclass
class SaveConfig:
    save_steps = []
    module_pahts = []


class MyAttnProcessor:
    saved_tensors: MultiDict2
    do_st_extended_attention = False
    do_st_averaged_attention = False
    target_frame_indices = [0]

    cur_timestep = 0
    cur_timestep_idx = 0

    is_cross_attn: bool
    is_self_attn: bool
    module_path: str

    save_cfg: SaveConfig

    def __init__(self, unet, unet_chunk_size=2):
        """
        :param unet_chunk_size:
            number of batches for each generated image, 2 for classifier free guidance
        """
        self.unet = unet
        self.unet_chunk_size = unet_chunk_size

    def call_init(self, attn, encoder_hidden_states):
        self.is_cross_attn = encoder_hidden_states is not None
        self.is_self_attn = not self.is_cross_attn
        self.module_path = get_module_path(self.unet, attn)

    def get_kv_hidden_states(
        self,
        attn: Attention,
        hidden_states: Float[torch.Tensor, "b t d"],
        encoder_hidden_states: Float[torch.Tensor, "b t d"],
    ) -> Float[torch.Tensor, "b t d"]:
        # cross attention
        if encoder_hidden_states is not None:
            hidden_states = encoder_hidden_states
            if attn.norm_cross is not None:
                hidden_states = attn.norm_cross(hidden_states)
            return hidden_states

        n_frames = hidden_states.shape[0] // self.unet_chunk_size

        if self.do_st_extended_attention:
            ext_hidden_states = extended_attn_kv_hidden_states(
                hidden_states,
                chunk_size=self.unet_chunk_size,
                frame_indices=self.target_frame_indices,
            )
            ext_hidden_states = extend_across_frame_dim(ext_hidden_states, n_frames)
            return ext_hidden_states

        if self.do_st_averaged_attention:
            ext_hidden_states = averaged_attn_kv_hidden_states(
                hidden_states,
                chunk_size=self.unet_chunk_size,
                frame_indices=self.target_frame_indices,
            )
            ext_hidden_states = extend_across_frame_dim(ext_hidden_states, n_frames)
            return ext_hidden_states

        return hidden_states

    def save_tensor(self, name: str, tensor: torch.Tensor):
        if self.cur_timestep_idx not in self.save_cfg.save_steps:
            return

        if self.module_path not in self.save_cfg.module_pahts:
            return

        keys = {
            "layer": self.module_path,
            "timestep": self.cur_timestep_idx,
            "name": name,
        }

        self.saved_tensors[keys] = tensor.cpu()

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

        # populate call-specific data fields
        self.call_init(attn, encoder_hidden_states)

        query = attn.to_q(hidden_states)

        kv_x = self.get_kv_hidden_states(attn, hidden_states, encoder_hidden_states)
        key = attn.to_k(kv_x)
        value = attn.to_v(kv_x)

        if self.is_self_attn:
            self.save_tensor("x", hidden_states)
            self.save_tensor("query", query)
            self.save_tensor("key", key)
            self.save_tensor("value", value)

        # compute attention
        attn_out = memory_efficient_attention(attn, key, query, value, attention_mask)

        # linear proj to output dim
        attn_out = attn.to_out[0](attn_out)
        attn_out = attn.to_out[1](attn_out)

        return attn_out
