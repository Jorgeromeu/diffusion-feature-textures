from enum import Enum
from math import sqrt
from typing import Dict, Optional

import torch
from diffusers.models.attention_processor import Attention
from einops import rearrange
from jaxtyping import Float
from rerun import Tensor

from text3d2video.attention_utils import (
    extend_across_frame_dim,
    extended_attn_kv_hidden_states,
    memory_efficient_attention,
)
from text3d2video.generative_rendering.configs import (
    GenerativeRenderingConfig,
)
from text3d2video.sd_feature_extraction import get_module_path
from text3d2video.util import blend_features


class AttentionMode(Enum):
    FEATURE_EXTRACTION: str = "extraction"
    FEATURE_INJECTION: str = "injection"


class GenerativeRenderingAttn:
    # generative rendering config
    gr_config: GenerativeRenderingConfig
    attn_mode: AttentionMode = AttentionMode.FEATURE_INJECTION

    # save features here
    saved_pre_attn: Dict[str, Float[torch.Tensor, "b t c"]] = {}
    saved_post_attn: Dict[str, Float[torch.Tensor, "b f t c"]] = {}

    post_attn_feature_images: Dict[str, Float[torch.Tensor, "b d h w"]] = {}

    module_path: str

    def __init__(self, unet, unet_chunk_size=2):
        """
        :param unet_chunk_size:
            number of batches for each generated image, 2 for classifier free guidance
        """
        self.unet = unet
        self.unet_chunk_size = unet_chunk_size

    def clear_saved_features(self):
        self.saved_pre_attn = {}
        self.saved_post_attn = {}

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

        if self.attn_mode == AttentionMode.FEATURE_EXTRACTION:
            # concatenate hidden states from all frames into one
            ext_hidden_states = extended_attn_kv_hidden_states(
                hidden_states, chunk_size=self.unet_chunk_size
            )

            # save pre_attn features
            self.save_pre_attn_features(ext_hidden_states)

            # repeat along frame-dimension
            return extend_across_frame_dim(ext_hidden_states, n_frames)

        if self.attn_mode == AttentionMode.FEATURE_INJECTION:
            saved_hidden_states = self.saved_pre_attn.get(module_path)

            # if no pre_attn features passed, use current hidden states
            if saved_hidden_states is None or not self.gr_config.do_pre_attn_injection:
                return hidden_states

            # repeat across frame dimension
            saved_hidden_states = extend_across_frame_dim(saved_hidden_states, n_frames)

            if self.gr_config.attend_to_self_kv:
                # concatenate current hidden states
                hidden_states = torch.cat([hidden_states, saved_hidden_states], dim=1)

            return saved_hidden_states

        return hidden_states

    def post_attn_injection(self, attn_out_square):
        # get feature images for current module
        feature_images = self.post_attn_feature_images.get(self.module_path)

        # skip blending
        if feature_images is None or not self.gr_config.do_post_attn_injection:
            return attn_out_square

        # blend rendered and current features
        blended = blend_features(
            attn_out_square,
            feature_images.to(attn_out_square),
            self.gr_config.feature_blend_alpha,
            channel_dim=2,
        )

        return blended

    def save_post_attn_features(self, attn_out_square: Tensor):
        save_features = (
            self.module_path in self.gr_config.module_paths
            and self.gr_config.do_post_attn_injection,
        )

        if save_features:
            self.saved_post_attn[self.module_path] = attn_out_square

    def save_pre_attn_features(self, hidden_states: Tensor):
        # save hidden states for later use
        save_pre_attn_features = (
            self.attn_mode
            and self.module_path in self.gr_config.module_paths
            and self.gr_config.do_pre_attn_injection
        )

        if save_pre_attn_features:
            self.saved_pre_attn[self.module_path] = hidden_states

    def call_init(self, attn: Attention):
        self.module_path = get_module_path(self.unet, attn)

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

        self.call_init(attn)

        query = attn.to_q(hidden_states)

        # get hidden states for k/v computation
        kv_hidden_states = self.get_kv_hidden_states(
            attn, self.module_path, hidden_states, encoder_hidden_states
        )

        key = attn.to_k(kv_hidden_states)
        value = attn.to_v(kv_hidden_states)

        # compute attention
        attn_out = memory_efficient_attention(attn, key, query, value, attention_mask)

        # reshape to square
        attn_out_square = rearrange(
            attn_out,
            "(b f) (h w) d -> b f d h w",
            h=int(sqrt(attn_out.shape[1])),
            b=self.unet_chunk_size,
        )

        # save post-attn features
        if self.attn_mode == AttentionMode.FEATURE_EXTRACTION:
            self.save_post_attn_features(attn_out_square)
        elif self.attn_mode == AttentionMode.FEATURE_INJECTION:
            attn_out_square = self.post_attn_injection(attn_out_square)

        # reshape back to 2d
        attn_out = rearrange(attn_out_square, "b f d h w -> (b f) (h w) d")

        # linear proj to output dim
        attn_out = attn.to_out[0](attn_out)
        attn_out = attn.to_out[1](attn_out)

        return attn_out
