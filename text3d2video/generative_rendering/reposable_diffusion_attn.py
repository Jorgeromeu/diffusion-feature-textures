from enum import Enum
from math import sqrt
from typing import Dict, Optional

import torch
from diffusers.models.attention_processor import Attention
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

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
    gr_data_artifact: GrDataArtifact

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

    def should_extract_pre_attn(self):
        return (
            self._cur_module_path in self.gr_config.module_paths
            and self.gr_config.do_pre_attn_injection
        )

    def should_extract_post_attn(self):
        return (
            self._cur_module_path in self.gr_config.module_paths
            and self.gr_config.do_post_attn_injection
        )

    def extract_pre_attn_features(self, kv_hidden_states: Tensor):
        self.pre_attn_features[self._cur_module_path] = kv_hidden_states

    def extract_post_attn_features(self, attn_out_square: Tensor):
        self.post_attn_features[self._cur_module_path] = attn_out_square

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

        if self.should_extract_pre_attn():
            self.extract_pre_attn_features(ext_hidden_states)

        kv_hidden_states = extend_across_frame_dim(ext_hidden_states, n_frames)

        key = attn.to_k(kv_hidden_states)
        value = attn.to_v(kv_hidden_states)
        qry = attn.to_q(hidden_states)

        attn_out = memory_efficient_attention(attn, key, qry, value, attention_mask)

        if self.should_extract_post_attn():
            attn_out_square = rearrange(
                attn_out,
                "(b f) (h w) d -> b f d h w",
                h=int(sqrt(attn_out.shape[1])),
                b=self.unet_chunk_size,
            )
            self.extract_post_attn_features(attn_out_square)

        return attn_out

    def call_injection_mode(
        self, attn: Attention, hidden_states: Tensor, attention_mask: Optional[Tensor]
    ):
        """
        Use injected key/value features and rendered post attention features
        """

        n_frames = hidden_states.shape[0] // self.unet_chunk_size
        qry = attn.to_q(hidden_states)

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

        attn_out = memory_efficient_attention(attn, key, qry, val, attention_mask)

        attn_out_square = rearrange(
            attn_out,
            "(b f) (h w) d -> b f d h w",
            h=int(sqrt(attn_out.shape[1])),
            b=self.unet_chunk_size,
        )

        # read injected post attn features
        injected_post_attn_features = self.post_attn_features.get(self._cur_module_path)

        if injected_post_attn_features is not None:
            # blend rendered and current features
            attn_out_square = blend_features(
                attn_out_square,
                injected_post_attn_features.to(attn_out_square),
                self.gr_config.feature_blend_alpha,
                channel_dim=2,
            )

        # reshape back to 2d
        attn_out = rearrange(attn_out_square, "b f d h w -> (b f) (h w) d")

        return attn_out

    def call_self_attn(self, attn, hidden_states, attention_mask):
        if self.mode == ReposableDiffusionAttnMode.FEATURE_EXTRACTION:
            attn_out = self.call_extraction_mode(attn, hidden_states, attention_mask)
        elif self.mode == ReposableDiffusionAttnMode.FEATURE_INJECTION:
            attn_out = self.call_injection_mode(attn, hidden_states, attention_mask)

        return attn_out
