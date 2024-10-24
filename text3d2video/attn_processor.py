from typing import Dict, Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from einops import rearrange
from jaxtyping import Float

from text3d2video.attention_utils import memory_efficient_attention
from text3d2video.sd_feature_extraction import get_module_path


class MyAttnProcessor:
    saved_tensors: Dict[str, torch.Tensor] = {}
    do_st_attention = True
    do_fst_attention = False

    is_cross_attn: bool
    is_self_attn: bool
    module_path: str

    def __init__(self, unet, unet_chunk_size=2):
        """
        :param unet_chunk_size:
            number of batches for each generated image, 2 for classifier free guidance
        """
        self.unet = unet
        self.unet_chunk_size = unet_chunk_size

    def memory_efficient_attention(self, attn, key, query, value, attention_mask):
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

        # pylint: disable=not-callable
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        attn_out = hidden_states.to(query.dtype)

        return attn_out

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

        if self.do_st_attention:
            # expand batch-frame dim, and stack all frames across seq dimension
            ext_hidden_states = rearrange(hidden_states, "(b f) t c -> b 1 (f t) c", f=n_frames)

            # repeat along frame dimension
            ext_hidden_states_repeated = ext_hidden_states.expand(-1, n_frames, -1, -1)

            # flatten batch-frame dimensions
            ext_hidden_states_repeated = rearrange(
                ext_hidden_states_repeated, "b f t d -> (b f) t d", f=n_frames
            )

            return ext_hidden_states_repeated

        if self.do_fst_attention:
            # expand batch-frame dim
            hidden_states_unstacked = rearrange(hidden_states, "(b f) t c -> b f t c", f=n_frames)

            # duplicate first frame for all frames
            frame_index = [0] * n_frames
            hidden_states_unstacked = hidden_states_unstacked[:, frame_index].detach().clone()

            # flatten batch-frame dimensions
            hidden_states_stacked = rearrange(hidden_states_unstacked, "b f t c -> (b f) t c")

            return hidden_states_stacked

        return hidden_states

    def save_tensor(self, name: str, tensor: torch.Tensor):
        self.saved_tensors[f"{self.module_path},{name}"] = tensor

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
            self.save_tensor("query", query)
            self.save_tensor("key", key)
            self.save_tensor("value", value)

        # compute attention
        attn_out = memory_efficient_attention(attn, key, query, value, attention_mask)

        # linear proj to output dim
        attn_out = attn.to_out[0](attn_out)
        attn_out = attn.to_out[1](attn_out)

        return attn_out
