from enum import Enum
from math import sqrt
from typing import Dict, Optional

import torch
from diffusers.models.attention_processor import Attention
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from text3d2video.adain import adain_2D
from text3d2video.artifacts.gr_data import GrDataArtifact
from text3d2video.attn_processor import DefaultAttnProcessor
from text3d2video.generative_rendering.configs import (
    GenerativeRenderingConfig,
    ReposableDiffusionConfig,
)
from text3d2video.util import blend_features
from text3d2video.utilities.attention_utils import (
    extend_across_frame_dim,
    extended_attn_kv_hidden_states,
    memory_efficient_attention,
)


class ReposableDiffusionAttnMode(Enum):
    FEATURE_EXTRACTION: str = "extraction"
    FEATURE_INJECTION: str = "injection"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False


class ReposableDiffusionAttn(DefaultAttnProcessor):
    gr_config: ReposableDiffusionConfig
    mode: ReposableDiffusionAttnMode = ReposableDiffusionAttnMode.FEATURE_INJECTION

    # save features here in extraction mode, and use them in injection mode
    pre_attn_features: Dict[str, Float[Tensor, "b t c"]] = {}
    post_attn_features: Dict[str, Float[Tensor, "b f t c"]] = {}

    def __init__(
        self,
        unet,
        gr_config: GenerativeRenderingConfig,
        mode=ReposableDiffusionAttnMode.FEATURE_INJECTION,
        unet_chunk_size=2,
    ):
        DefaultAttnProcessor.__init__(self, unet, unet_chunk_size)
        self.gr_config = gr_config
        self.mode = mode
        self.pre_attn_features = {}
        self.post_attn_features = {}

    def should_extract(self):
        return self._cur_module_path in self.gr_config.module_paths

    # functionality

    def call_extraction_mode(
        self, attn: Attention, hidden_states: Tensor, attention_mask: Optional[Tensor]
    ):
        """
        Perform extended attention, and extract pre and post attention feature
        """

        n_frames = hidden_states.shape[0] // self.unet_chunk_size

        ext_hidden_states = extended_attn_kv_hidden_states(
            hidden_states, chunk_size=self.unet_chunk_size
        )

        if self.should_extract():
            self.pre_attn_features[self._cur_module_path] = ext_hidden_states

        kv_hidden_states = extend_across_frame_dim(ext_hidden_states, n_frames)

        key = attn.to_k(kv_hidden_states)
        value = attn.to_v(kv_hidden_states)
        qry = attn.to_q(hidden_states)

        attn_out = memory_efficient_attention(attn, key, qry, value, attention_mask)

        if self.should_extract():
            if self.gr_config.aggregate_queries:
                spatial_features = qry
            else:
                spatial_features = attn_out

            height = int(sqrt(spatial_features.shape[1]))

            qry_square = rearrange(
                spatial_features,
                "(b f) (h w) d -> b f d h w",
                h=height,
                b=self.unet_chunk_size,
            )
            self.post_attn_features[self._cur_module_path] = qry_square

        return attn_out

    def call_injection_mode(
        self, attn: Attention, hidden_states: Tensor, attention_mask: Optional[Tensor]
    ):
        """
        Use injected key/value features and rendered post attention features
        """

        def blend_feature_images(original_features_1D: Tensor, feature_images: Tensor):
            height = int(sqrt(original_features_1D.shape[1]))
            original_features_2D = rearrange(
                original_features_1D,
                "(b f) (h w) d -> b f d h w",
                h=height,
                b=self.unet_chunk_size,
            )

            # blend rendered and current features
            blended = blend_features(
                original_features_2D,
                feature_images.to(original_features_2D),
                self.gr_config.feature_blend_alpha,
                channel_dim=2,
            )

            # reshape back to 2D
            blended_features_1D = rearrange(blended, "b f d h w -> (b f) (h w) d")

            return blended_features_1D

        n_frames = hidden_states.shape[0] // self.unet_chunk_size

        injected_kv_features = self.pre_attn_features.get(self._cur_module_path)

        # if nothing passed, do normal self attention
        if injected_kv_features is None:
            kv_features = hidden_states

        # if passed attend to injected kv/features
        else:
            kv_features = extend_across_frame_dim(injected_kv_features, n_frames)

            # concatenate current hidden states
            if self.gr_config.attend_to_self_kv:
                kv_features = torch.cat([hidden_states, kv_features], dim=1)

        key = attn.to_k(kv_features)
        val = attn.to_v(kv_features)
        qry = attn.to_q(hidden_states)

        # read injected post attn features
        injected_spatial_features = self.post_attn_features.get(self._cur_module_path)

        if injected_spatial_features is not None and self.gr_config.aggregate_queries:
            qry = blend_feature_images(qry, injected_spatial_features)

        # self.write_qkv(qry, key, val)

        attn_out = memory_efficient_attention(attn, key, qry, val, attention_mask)

        if (
            injected_spatial_features is not None
            and not self.gr_config.aggregate_queries
        ):
            attn_out = blend_feature_images(attn_out, injected_spatial_features)

        return attn_out

    def call_cross_attn(
        self, attn, hidden_states, encoder_hidden_states, attention_mask
    ):
        qry = attn.to_q(hidden_states)

        kv_hidden_states = encoder_hidden_states
        if attn.norm_cross is not None:
            kv_hidden_states = attn.norm_cross(hidden_states)

        key = attn.to_k(kv_hidden_states)
        val = attn.to_v(kv_hidden_states)

        if self.mode == ReposableDiffusionAttnMode.FEATURE_INJECTION:
            self.write_qkv(qry, key, val)

        return memory_efficient_attention(attn, key, qry, val, attention_mask)

    def call_self_attn(self, attn, hidden_states, attention_mask):
        if self.mode == ReposableDiffusionAttnMode.FEATURE_EXTRACTION:
            attn_out = self.call_extraction_mode(attn, hidden_states, attention_mask)
        elif self.mode == ReposableDiffusionAttnMode.FEATURE_INJECTION:
            attn_out = self.call_injection_mode(attn, hidden_states, attention_mask)

        return attn_out
