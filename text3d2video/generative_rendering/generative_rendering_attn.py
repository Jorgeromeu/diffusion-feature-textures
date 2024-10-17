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


class GenerativeRenderingAttn:

    # rerun
    rerun: bool = False
    rerun_module_paths = []

    # modules to save/inject features
    module_paths = []

    # types of attention
    do_extended_attention: bool = False
    do_pre_attn_injection: bool = False

    # wether or not to save features
    save_pre_attn_features = False
    save_post_attn_features = False
    save_blended_features = False

    saved_pre_attn: Dict[str, Float[torch.Tensor, "b t c"]] = {}
    saved_post_attn: Dict[str, Float[torch.Tensor, "b f t c"]] = {}

    # post attention
    do_post_attn_injection: bool = False
    feature_blend_alpha: float = 1
    feature_images: Dict[str, Float[torch.Tensor, "b d h w"]] = {}

    chunk_indices: List[int] = []

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
        module_path: str,
        hidden_states: Float[torch.Tensor, "b t d"],
        encoder_hidden_states: Float[torch.Tensor, "b t d"],
    ) -> Float[torch.Tensor, "b t d"]:

        # if encoder hidden states are provided use them for cross attention
        if encoder_hidden_states is not None:
            hidden_states = encoder_hidden_states
            if attn.norm_cross is not None:
                hidden_states = attn.norm_cross(hidden_states)
            return hidden_states

        n_frames = hidden_states.shape[0] // self.unet_chunk_size

        if self.do_extended_attention:

            # stack all frames across time dimension
            # unet_chunk_size * (n_frames * sequence_length) hidden_size
            ext_hidden_states = rearrange(
                hidden_states, "(b f) t c -> b (f t) c", f=n_frames
            )

            # save hidden states for later use
            if self.save_pre_attn_features and module_path in self.module_paths:
                self.saved_pre_attn[module_path] = ext_hidden_states

            # repeat along batch dimension
            # repeat b t c -> (b f) t c
            ext_hidden_states = (
                ext_hidden_states.unsqueeze(1)
                .expand(-1, n_frames, -1, -1)
                .reshape(
                    -1,
                    ext_hidden_states.shape[1],
                    ext_hidden_states.shape[2],
                )
            )

            return ext_hidden_states

        if self.do_pre_attn_injection:

            # get saved hidden states
            saved_hidden_states = self.saved_pre_attn.get(module_path)
            if saved_hidden_states is None:
                return hidden_states

            # repeat saved hdiden states along batch dimension
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

            # concatenate saved hidden states with current hidden states
            # (b f) t*(f+1) c
            concated_hidden_states = torch.cat(
                [saved_hidden_states, hidden_states], dim=1
            )

            return concated_hidden_states

        return hidden_states

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
        module_path = get_module_path(self.unet, attn)

        query = attn.to_q(hidden_states)

        # get hidden states for k/v computation
        kv_hidden_states = self.get_kv_hidden_states(
            attn, module_path, hidden_states, encoder_hidden_states
        )

        key = attn.to_k(kv_hidden_states)
        value = attn.to_v(kv_hidden_states)

        # compute attention
        attn_out = self.memory_efficient_attention(
            attn, key, query, value, attention_mask
        )

        # reshape to square
        attn_out_square = rearrange(
            attn_out,
            "(b f) (h w) d -> b f d h w",
            h=int(sqrt(attn_out.shape[1])),
            b=self.unet_chunk_size,
        )

        # save post attention features
        if self.save_post_attn_features and module_path in self.module_paths:
            self.saved_post_attn[module_path] = attn_out_square

        # inject features
        feature_images = self.feature_images.get(module_path)
        if self.do_post_attn_injection and feature_images is not None:
            # blend rendered and current features
            blended = blend_features(
                attn_out_square,
                feature_images.to(attn_out),
                self.feature_blend_alpha,
                channel_dim=2,
            )

            attn_out_square = blended

        # log features
        if module_path in self.rerun_module_paths and self.rerun:

            attn_out_frame = attn_out_square[0, 0, :, :, :]
            rendered_frame = feature_images[0, 0, :, :, :]
            blended_frame = blended[0, 0, :, :, :]

            rr.log(
                f"attn_out",
                rr.Image(self.pca.feature_map_to_rgb_pil(attn_out_frame.cpu())),
            )

            rr.log(
                f"rendered",
                rr.Image(self.pca.feature_map_to_rgb_pil(rendered_frame.cpu())),
            )

            rr.log(
                f"blended",
                rr.Image(self.pca.feature_map_to_rgb_pil(blended_frame.cpu())),
            )

        attn_out = rearrange(attn_out_square, "b f d h w -> (b f) (h w) d")

        # linear proj to output dim
        attn_out = attn.to_out[0](attn_out)
        attn_out = attn.to_out[1](attn_out)

        return attn_out
