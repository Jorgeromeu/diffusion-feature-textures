from enum import Enum
from math import sqrt
from typing import Dict, Optional

import torch
from diffusers.models.attention_processor import Attention
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from text3d2video.artifacts.gr_data import GrDataArtifact
from text3d2video.generative_rendering.configs import (
    GenerativeRenderingConfig,
)
from text3d2video.sd_feature_extraction import get_module_path
from text3d2video.util import blend_features
from text3d2video.utilities.attention_utils import (
    extend_across_frame_dim,
    extended_attn_kv_hidden_states,
    memory_efficient_attention,
)


class GrAttnMode(Enum):
    FEATURE_EXTRACTION: str = "extraction"
    FEATURE_INJECTION: str = "injection"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False


class GenerativeRenderingAttn:
    # attn processor config
    gr_config: GenerativeRenderingConfig
    mode: GrAttnMode = GrAttnMode.FEATURE_INJECTION

    # gr data
    gr_data_artifact: GrDataArtifact

    chunk_frame_indices: Tensor

    # save features here in extraction mode, and use them in injection mode
    pre_attn_features: Dict[str, Float[Tensor, "b t c"]] = {}
    post_attn_features: Dict[str, Float[Tensor, "b f t c"]] = {}

    # state variables
    cur_timestep = 0
    _cur_module_path: str
    _is_cross_attn: bool
    _is_self_attn: bool

    def __init__(
        self,
        unet,
        gr_config: GenerativeRenderingConfig,
        unet_chunk_size=2,
        mode=GrAttnMode.FEATURE_INJECTION,
    ):
        """
        :param unet_chunk_size:
            number of batches for each generated image, 2 for classifier free guidance
        """
        self.unet = unet
        self.unet_chunk_size = unet_chunk_size
        self.gr_config = gr_config
        self.mode = mode

        self.pre_attn_features = {}
        self.post_attn_features = {}

    def should_save_pre_attn(self):
        return (
            self._cur_module_path in self.gr_config.module_paths
            and self.gr_config.do_pre_attn_injection
        )

    def should_save_post_attn(self):
        return (
            self._cur_module_path in self.gr_config.module_paths
            and self.gr_config.do_post_attn_injection
        )

    def save_pre_attn_features(self, kv_hidden_states: Tensor):
        self.pre_attn_features[self._cur_module_path] = kv_hidden_states

    def save_post_attn_features(self, attn_out_square: Tensor):
        self.post_attn_features[self._cur_module_path] = attn_out_square

    # functionality

    def call_init(self, attn: Attention, encoder_hidden_states: Tensor):
        self._cur_module_path = get_module_path(self.unet, attn)
        self._is_cross_attn = encoder_hidden_states is not None
        self._is_self_attn = not self._is_cross_attn

    def call_cross_attn(
        self,
        attn: Attention,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        attention_mask: Optional[Tensor],
    ):
        qry = attn.to_q(hidden_states)

        kv_hidden_states = encoder_hidden_states
        if attn.norm_cross is not None:
            kv_hidden_states = attn.norm_cross(hidden_states)

        key = attn.to_k(kv_hidden_states)
        value = attn.to_v(kv_hidden_states)

        return memory_efficient_attention(attn, key, qry, value, attention_mask)

    def call_extraction(
        self, attn: Attention, hidden_states: Tensor, attention_mask: Optional[Tensor]
    ):
        qry = attn.to_q(hidden_states)

        n_frames = hidden_states.shape[0] // self.unet_chunk_size

        if self.should_save_pre_attn():
            # concatenate hidden states from all frames into one
            ext_hidden_states = extended_attn_kv_hidden_states(
                hidden_states, chunk_size=self.unet_chunk_size
            )

            # save pre_attn features
            self.save_pre_attn_features(ext_hidden_states)

            kv_hidden_states = extend_across_frame_dim(ext_hidden_states, n_frames)
        else:
            kv_hidden_states = hidden_states

        key = attn.to_k(kv_hidden_states)
        value = attn.to_v(kv_hidden_states)

        attn_out = memory_efficient_attention(attn, key, qry, value, attention_mask)

        if self.should_save_post_attn():
            attn_out_square = rearrange(
                attn_out,
                "(b f) (h w) d -> b f d h w",
                h=int(sqrt(attn_out.shape[1])),
                b=self.unet_chunk_size,
            )
            self.save_post_attn_features(attn_out_square)

        return attn_out

    def call_injection(
        self, attn: Attention, hidden_states: Tensor, attention_mask: Optional[Tensor]
    ):
        n_frames = hidden_states.shape[0] // self.unet_chunk_size
        qry = attn.to_q(hidden_states)

        # read injected features
        injected_hidden_states = self.pre_attn_features.get(self._cur_module_path)

        # if nothing passed, do normal self attention
        if injected_hidden_states is None:
            kv_hidden_states = hidden_states
        else:
            # repeat across frame dimension
            kv_hidden_states = extend_across_frame_dim(injected_hidden_states, n_frames)

            if self.gr_config.attend_to_self_kv:
                # concatenate current hidden states
                kv_hidden_states = torch.cat([hidden_states, kv_hidden_states], dim=1)

        key = attn.to_k(kv_hidden_states)
        val = attn.to_v(kv_hidden_states)

        # save pre attn features
        unstacked_q = rearrange(qry, "(b f) t c -> b f t c", f=n_frames)
        unstacked_k = rearrange(key, "(b f) t c -> b f t c", f=n_frames)
        unstacked_v = rearrange(val, "(b f) t c -> b f t c", f=n_frames)
        self.gr_data_artifact.attn_writer.write_qkv_batched(
            self.cur_timestep,
            self._cur_module_path,
            unstacked_q,
            unstacked_k,
            unstacked_v,
            chunk_frame_indices=self.chunk_frame_indices,
        )

        attn_out = memory_efficient_attention(attn, key, qry, val, attention_mask)

        attn_out_square = rearrange(
            attn_out,
            "(b f) (h w) d -> b f d h w",
            h=int(sqrt(attn_out.shape[1])),
            b=self.unet_chunk_size,
        )

        # write post_attn features before injection
        self.gr_data_artifact.gr_writer.write_post_attn_pre_injection(
            self.cur_timestep,
            self._cur_module_path,
            attn_out_square,
            self.chunk_frame_indices,
        )

        # read injected post attn features
        injected_post_attn_features = self.post_attn_features.get(self._cur_module_path)

        if injected_post_attn_features is not None:
            # save injected post attn features
            self.gr_data_artifact.gr_writer.write_post_attn_renders(
                self.cur_timestep,
                self._cur_module_path,
                injected_post_attn_features,
                self.chunk_frame_indices,
            )

            # blend rendered and current features
            attn_out_square = blend_features(
                attn_out_square,
                injected_post_attn_features.to(attn_out_square),
                self.gr_config.feature_blend_alpha,
                channel_dim=2,
            )

            # save post injection features
            self.gr_data_artifact.gr_writer.write_post_attn_post_injection(
                self.cur_timestep,
                self._cur_module_path,
                attn_out_square,
                self.chunk_frame_indices,
            )

        # reshape back to 2d
        attn_out = rearrange(attn_out_square, "b f d h w -> (b f) (h w) d")

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

        self.call_init(attn, encoder_hidden_states)

        if self._is_cross_attn:
            attn_out = self.call_cross_attn(
                attn, hidden_states, encoder_hidden_states, attention_mask
            )

        elif self.mode == GrAttnMode.FEATURE_EXTRACTION:
            attn_out = self.call_extraction(attn, hidden_states, attention_mask)

        elif self.mode == GrAttnMode.FEATURE_INJECTION:
            attn_out = self.call_injection(attn, hidden_states, attention_mask)

        else:
            query = attn.to_q(hidden_states)

            # get hidden states for k/v computation
            kv_hidden_states = self.get_kv_hidden_states(
                attn, self._cur_module_path, hidden_states, encoder_hidden_states
            )

            key = attn.to_k(kv_hidden_states)
            value = attn.to_v(kv_hidden_states)

            # compute attention
            attn_out = memory_efficient_attention(
                attn, key, query, value, attention_mask
            )

            # reshape to square
            attn_out_square = rearrange(
                attn_out,
                "(b f) (h w) d -> b f d h w",
                h=int(sqrt(attn_out.shape[1])),
                b=self.unet_chunk_size,
            )

            # save post-attn features
            if (
                self.mode == GrAttnMode.FEATURE_EXTRACTION
                and self.should_save_post_attn()
            ):
                self.save_post_attn_features(attn_out_square)

            if self.mode == GrAttnMode.FEATURE_INJECTION:
                # save qkv

                # unstack batch-frame dimension
                n_frames = hidden_states.shape[0] // self.unet_chunk_size
                unstacked_q = rearrange(query, "(b f) t c -> b f t c", f=n_frames)
                unstacked_k = rearrange(key, "(b f) t c -> b f t c", f=n_frames)
                unstacked_v = rearrange(value, "(b f) t c -> b f t c", f=n_frames)

                self.gr_data_artifact.attn_writer.write_qkv_batched(
                    self.cur_timestep,
                    self._cur_module_path,
                    unstacked_q,
                    unstacked_k,
                    unstacked_v,
                    chunk_frame_indices=self.chunk_frame_indices,
                )
                # save pre_injection features
                self.gr_data_artifact.gr_writer.write_post_attn_pre_injection(
                    self.cur_timestep,
                    self._cur_module_path,
                    attn_out_square,
                    self.chunk_frame_indices,
                )
                attn_out_square = self.post_attn_injection(attn_out_square)
                # save post injection features
                self.gr_data_artifact.gr_writer.write_post_attn_post_injection(
                    self.cur_timestep,
                    self._cur_module_path,
                    attn_out_square,
                    self.chunk_frame_indices,
                )

            # reshape back to 2d
            attn_out = rearrange(attn_out_square, "b f d h w -> (b f) (h w) d")

        # linear proj to output dim
        attn_out = attn.to_out[0](attn_out)
        attn_out = attn.to_out[1](attn_out)

        return attn_out
