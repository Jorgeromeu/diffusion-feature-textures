from typing import Dict, Optional
import torch
from einops import einsum, rearrange, repeat
from diffusers.models.attention_processor import Attention

from text3d2video.multidict import MultiDict
from text3d2video.sd_feature_extraction import get_module_path
import torch.nn.functional as F


class GenerativeRenderingAttn:

    # perform extended attention
    do_extended_attention: bool = False
    do_cross_frame_attn: bool = False

    # use saved inputs for key and value computation
    saved_pre_attn: Dict[str, torch.Tensor]
    do_pre_attn_injection: bool = False

    # use saved post_attn output for attention computation
    saved_post_attn: torch.Tensor = None
    do_post_attn_injection: bool = False

    def __init__(self, unet, unet_chunk_size=2):
        """
        :param unet_chunk_size:
            number of batches for each generated image, 2 for classifier free guidance
        """
        self.unet = unet
        self.unet_chunk_size = unet_chunk_size

    def do_memory_efficient_attention(self, attn, key, query, value, attention_mask):

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

        # hidden_states: (batch_size, sequence_length, c)
        batch_size, sequence_length, _ = hidden_states.shape
        n_frames = batch_size // self.unet_chunk_size

        query = attn.to_q(hidden_states)

        is_cross_attention = encoder_hidden_states is not None
        is_self_attention = not is_cross_attention

        # if cross attention, use encoder_hidden_states for key and value
        if is_cross_attention:
            hidden_states = encoder_hidden_states
            if attn.norm_cross is not None:
                hidden_states = attn.norm_cross(hidden_states)

        # project to key, value
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Cross-frame attention
        if is_self_attention and self.do_cross_frame_attn:

            frame_index = [0] * n_frames

            # for key and value, use frame 0
            key = rearrange(key, "(b f) d c -> b f d c", f=n_frames)
            key = key[:, frame_index].detach().clone()
            key = rearrange(key, "b f d c -> (b f) d c")

            value = rearrange(value, "(b f) d c -> b f d c", f=n_frames)
            value = value[:, frame_index].detach().clone()
            value = rearrange(value, "b f d c -> (b f) d c")

        # Extended attention
        if is_self_attention and self.do_extended_attention:

            # stack all frames across time dimension
            hidden_states_extended = rearrange(
                hidden_states, "(b f) t c -> b (f t) c", f=n_frames
            )

            # hidden_states_extended = repeat(
            #     hidden_states_extended, "b t c -> (b f) t c", f=n_frames
            # )
            hidden_states_extended = (
                hidden_states_extended.unsqueeze(1)
                .expand(-1, n_frames, -1, -1)
                .reshape(
                    -1, hidden_states_extended.shape[1], hidden_states_extended.shape[2]
                )
            )
            key = attn.to_k(hidden_states_extended)
            value = attn.to_v(hidden_states_extended)

        attn_out = self.do_memory_efficient_attention(
            attn, key, query, value, attention_mask
        )

        # linear proj to output dim
        attn_out = attn.to_out[0](attn_out)
        attn_out = attn.to_out[1](attn_out)

        return attn_out
