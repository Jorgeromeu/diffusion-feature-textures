from math import sqrt
from typing import Dict, List, Optional

import rerun as rr
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from einops import rearrange
from jaxtyping import Float

from text3d2video.sd_feature_extraction import get_module_path
from text3d2video.util import blend_features


class MyAttnProcessor:

    saved_tensors: Dict[str, torch.Tensor] = {}

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

        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, attn.heads * head_dim
        )
        attn_out = hidden_states.to(query.dtype)

        return attn_out

    def get_kv_hidden_states(
        self,
        attn: Attention,
        hidden_states: Float[torch.Tensor, "b t d"],
        encoder_hidden_states: Float[torch.Tensor, "b t d"],
    ) -> Float[torch.Tensor, "b t d"]:

        # if encoder hidden states are provided use them for cross attention
        if encoder_hidden_states is not None:
            hidden_states = encoder_hidden_states
            if attn.norm_cross is not None:
                hidden_states = attn.norm_cross(hidden_states)
            return hidden_states

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

        # figure out which attention layer we are in
        self.module_path = get_module_path(self.unet, attn)

        query = attn.to_q(hidden_states)

        # get hidden states for k/v
        kv_hidden_states = self.get_kv_hidden_states(
            attn, hidden_states, encoder_hidden_states
        )

        key = attn.to_k(kv_hidden_states)
        value = attn.to_v(kv_hidden_states)

        self.save_tensor("query", query)
        self.save_tensor("key", key)
        self.save_tensor("value", value)

        # compute attention
        attn_out = self.memory_efficient_attention(
            attn, key, query, value, attention_mask
        )

        # linear proj to output dim
        attn_out = attn.to_out[0](attn_out)
        attn_out = attn.to_out[1](attn_out)

        return attn_out
