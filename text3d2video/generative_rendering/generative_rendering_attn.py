from math import sqrt
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from einops import rearrange

from text3d2video.sd_feature_extraction import get_module_path


class GenerativeRenderingAttn:

    # perform extended attention
    do_extended_attention: bool = False
    do_cross_frame_attn: bool = False

    # use saved inputs for key and value computation
    do_pre_attn_injection: bool = False
    saved_pre_attn: Dict[str, torch.Tensor] = {}

    # use saved post_attn output for attention computation
    saved_post_attn: Dict[str, torch.Tensor] = {}
    feature_images: Dict[str, torch.Tensor] = {}
    do_post_attn_injection: bool = False

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
        batch_size, _, _ = hidden_states.shape
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

        if is_self_attention:

            # cross frame attention
            if self.do_cross_frame_attn:
                frame_index = [0] * n_frames

                # for key and value, use frame 0
                key = rearrange(key, "(b f) d c -> b f d c", f=n_frames)
                key = key[:, frame_index].detach().clone()
                key = rearrange(key, "b f d c -> (b f) d c")

                value = rearrange(value, "(b f) d c -> b f d c", f=n_frames)
                value = value[:, frame_index].detach().clone()
                value = rearrange(value, "b f d c -> (b f) d c")

            # Extended attention
            elif self.do_extended_attention:

                # stack all frames across time dimension
                # unet_chunk_size * (n_frames * sequence_length) hidden_size
                saved_hidden_states = rearrange(
                    hidden_states, "(b f) t c -> b (f t) c", f=n_frames
                )

                # save pre attn features
                module_path = get_module_path(self.unet, attn)
                self.saved_pre_attn[module_path] = saved_hidden_states

                # repeat b t c -> (b f) t c
                saved_hidden_states = (
                    saved_hidden_states.unsqueeze(1)
                    .expand(-1, n_frames, -1, -1)
                    .reshape(
                        -1,
                        saved_hidden_states.shape[1],
                        saved_hidden_states.shape[2],
                    )
                )

                key = attn.to_k(saved_hidden_states)
                value = attn.to_v(saved_hidden_states)

            # pre attn injection
            elif self.do_pre_attn_injection:

                # get saved hidden states
                module_path = get_module_path(self.unet, attn)
                saved_hidden_states = self.saved_pre_attn.get(module_path)

                # if exists, use saved hidden states for key and value computation
                if saved_hidden_states is not None:

                    # repeat b t c -> (b f) t c
                    saved_hidden_states = (
                        saved_hidden_states.unsqueeze(1)
                        .expand(-1, n_frames, -1, -1)
                        .reshape(
                            -1,
                            saved_hidden_states.shape[1],
                            saved_hidden_states.shape[2],
                        )
                    )

                    concated_hidden_states = torch.cat(
                        [saved_hidden_states, hidden_states], dim=1
                    )

                    key = attn.to_k(concated_hidden_states)
                    value = attn.to_v(concated_hidden_states)

        attn_out = self.memory_efficient_attention(
            attn, key, query, value, attention_mask
        )

        if self.do_post_attn_injection:
            module_path = get_module_path(self.unet, attn)
            feature_images = self.feature_images.get(module_path)

            if feature_images is not None:

                feature_res = int(sqrt(attn_out.shape[1]))
                attn_out_square = rearrange(
                    attn_out,
                    "(b f) (h w) c -> b f c h w",
                    b=self.unet_chunk_size,
                    h=feature_res,
                    w=feature_res,
                )

                print(module_path)
                print("feature_im", feature_images.shape)
                print("attn_out", attn_out_square.shape)

        self.saved_post_attn[get_module_path(self.unet, attn)] = attn_out

        # linear proj to output dim
        attn_out = attn.to_out[0](attn_out)
        attn_out = attn.to_out[1](attn_out)

        return attn_out
