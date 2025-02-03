import torch
from diffusers.models.attention_processor import Attention
from einops import rearrange
from torch import Tensor

from text3d2video.adain import adain_1D
from text3d2video.attn_processor import DefaultAttnProcessor
from text3d2video.utilities.attention_utils import (
    extend_across_frame_dim,
    extended_attn_kv_hidden_states,
    memory_efficient_attention,
)


class StyleAlignedAttentionProcessor(DefaultAttnProcessor):
    chunk_frame_indices: Tensor

    def __init__(
        self,
        unet,
        ref_index: int = 0,
        attend_to: str = "both",
        adain_self_features: bool = False,
        unet_chunk_size=2,
    ):
        """
        :param unet_chunk_size:
            number of batches for each generated image, 2 for classifier free guidance
        """
        self.adain_self_features = adain_self_features
        self.unet = unet
        self.ref_index = ref_index
        self.attend_to = attend_to
        self.unet_chunk_size = unet_chunk_size

    def call_self_attn(
        self, attn: Attention, hidden_states: Tensor, attention_mask: Tensor
    ):
        # get reference features
        n_frames = hidden_states.shape[0] // self.unet_chunk_size
        unstacked_x = rearrange(hidden_states, "(b f) t c -> b f t c", f=n_frames)
        x_ref = unstacked_x[:, self.ref_index, ...]
        x_ref = extend_across_frame_dim(x_ref, n_frames)

        key_self = attn.to_k(hidden_states)
        val_self = attn.to_v(hidden_states)

        key_ref = attn.to_k(x_ref)
        val_ref = attn.to_v(x_ref)
        qry_ref = attn.to_q(x_ref)
        qry_self = attn.to_q(hidden_states)

        if self.adain_self_features:
            qry_self = adain_1D(qry_self, qry_ref)
            key_self = adain_1D(key_self, key_ref)

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

        self.write_qkv(qry_self, key, val)

        return memory_efficient_attention(attn, key, qry_self, val, attention_mask)
