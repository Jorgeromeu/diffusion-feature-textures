from dataclasses import dataclass
from typing import Optional

import torch
from diffusers.models.attention_processor import Attention
from jaxtyping import Float

from text3d2video.attention_utils import (
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


class MultiFrameAttnProcessor:
    """
    Attention processor that supports sharing k/v sequences across generated frames
    """

    # tensor-saving fields
    save_cfg: SaveConfig
    saved_tensors: MultiDict2

    # config fields
    attend_to_self = True
    target_frame_indices = [0]

    # state variables
    cur_timestep = 0
    cur_timestep_idx = 0
    _is_cross_attn: bool
    _is_self_attn: bool
    _cur_module_path: str

    def __init__(self, unet, unet_chunk_size=2):
        """
        :param unet_chunk_size:
            number of batches for each generated image, 2 for classifier free guidance
        """
        self.unet = unet
        self.unet_chunk_size = unet_chunk_size

    def call_init(self, attn, encoder_hidden_states):
        self._is_cross_attn = encoder_hidden_states is not None
        self._is_self_attn = not self._is_cross_attn
        self._cur_module_path = get_module_path(self.unet, attn)

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

        b, _, d = hidden_states.shape
        kv_sequence = torch.empty(
            b, 0, d, device=hidden_states.device, dtype=hidden_states.dtype
        )

        # include self-features
        if self.attend_to_self:
            kv_sequence = torch.cat([kv_sequence, hidden_states], dim=1)

        n_frames = hidden_states.shape[0] // self.unet_chunk_size

        ext_hidden_states = extended_attn_kv_hidden_states(
            hidden_states,
            chunk_size=self.unet_chunk_size,
            frame_indices=self.target_frame_indices,
        )
        ext_hidden_states = extend_across_frame_dim(ext_hidden_states, n_frames)

        # include extended features
        kv_sequence = torch.cat([kv_sequence, ext_hidden_states], dim=1)

        return kv_sequence

    def save_tensor(self, name: str, tensor: torch.Tensor):
        if self.cur_timestep_idx not in self.save_cfg.save_steps:
            return

        if self._cur_module_path not in self.save_cfg.module_pahts:
            return

        keys = {
            "layer": self._cur_module_path,
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

        if self._is_self_attn:
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
