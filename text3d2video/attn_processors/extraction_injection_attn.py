from enum import Enum
from math import sqrt
from typing import Dict, List, Optional

import torch
from diffusers.models.attention_processor import Attention
from einops import rearrange
from jaxtyping import Float
from torch import Tensor

from text3d2video.attn_processors.attn_processor import DefaultAttnProcessor
from text3d2video.util import blend_features
from text3d2video.utilities.attention_utils import (
    extend_across_frame_dim,
    extended_attn_kv_hidden_states,
    memory_efficient_attention,
)


class AttnMode(Enum):
    FEATURE_EXTRACTION: str = "extraction"
    FEATURE_INJECTION: str = "injection"

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False


class ExtractionInjectionAttn(DefaultAttnProcessor):
    """
    General Purpose Attention processor that enables extracting and
    injecting features in attention layers
    """

    # extraction settings
    do_kv_extraction: bool
    do_spatial_qry_extraction: bool
    do_spatial_post_attn_extraction: bool
    kv_extraction_paths: List[str]
    spatial_qry_extraction_paths: List[str]
    spatial_post_attn_extraction_paths: List[str]

    # mode
    mode: AttnMode = AttnMode.FEATURE_INJECTION

    # save features here in extraction mode, and use them in injection mode
    kv_features: Dict[str, Float[Tensor, "b t c"]] = {}
    spatial_qry_features: Dict[str, Float[Tensor, "b f t c"]] = {}
    spatial_post_attn_features: Dict[str, Float[Tensor, "b f t c"]] = {}

    def __init__(
        self,
        model,
        do_spatial_qry_extraction: bool,
        do_spatial_post_attn_extraction: bool,
        do_kv_extraction: bool,
        attend_to_self_kv: bool,
        feature_blend_alpha: bool,
        kv_extraction_paths: List[str],
        spatial_qry_extraction_paths: List[str],
        spatial_post_attn_extraction_paths: List[str],
        mode=AttnMode.FEATURE_INJECTION,
        unet_chunk_size=2,
    ):
        DefaultAttnProcessor.__init__(self, model, unet_chunk_size)
        self.do_kv_extraction = do_kv_extraction
        self.do_spatial_qry_extraction = do_spatial_qry_extraction
        self.do_spatial_post_attn_extraction = do_spatial_post_attn_extraction
        self.attend_to_self_kv = attend_to_self_kv
        self.feature_blend_alpha = feature_blend_alpha
        self.kv_extraction_paths = kv_extraction_paths
        self.spatial_qry_extraction_paths = spatial_qry_extraction_paths
        self.spatial_post_attn_extraction_paths = spatial_post_attn_extraction_paths
        self.mode = mode
        self.kv_features = {}
        self.spatial_qry_features = {}

    def _should_extract_qrys(self):
        return (
            self.do_spatial_qry_extraction
            and self._cur_module_path in self.spatial_qry_extraction_paths
        )

    def _should_extract_post_attn(self):
        return (
            self.do_spatial_post_attn_extraction
            and self._cur_module_path in self.spatial_post_attn_extraction_paths
        )

    def _should_extract_kv(self):
        return (
            self.do_kv_extraction and self._cur_module_path in self.kv_extraction_paths
        )

    # public facing mode-setting methods

    def set_extraction_mode(self):
        self.mode = AttnMode.FEATURE_EXTRACTION
        self.clear_features()

    def clear_features(self):
        self.kv_features = {}
        self.spatial_qry_features = {}
        self.spatial_post_attn_features = {}

    def set_injection_mode(
        self, pre_attn_features={}, post_attn_features={}, qry_features={}
    ):
        self.mode = AttnMode.FEATURE_INJECTION
        self.kv_features = pre_attn_features
        self.spatial_post_attn_features = post_attn_features
        self.spatial_qry_features = qry_features

    # functionality

    def _call_extraction_mode(
        self, attn: Attention, hidden_states: Tensor, attention_mask: Optional[Tensor]
    ):
        """
        Perform extended attention, and extract features
        """

        n_frames = hidden_states.shape[0] // self.chunk_size

        ext_hidden_states = extended_attn_kv_hidden_states(
            hidden_states, chunk_size=self.chunk_size
        )

        kv_hidden_states = extend_across_frame_dim(ext_hidden_states, n_frames)

        key = attn.to_k(kv_hidden_states)
        value = attn.to_v(kv_hidden_states)
        qry = attn.to_q(hidden_states)

        attn_out = memory_efficient_attention(attn, key, qry, value, attention_mask)

        # extract kv-features
        if self._should_extract_kv():
            self.kv_features[self._cur_module_path] = ext_hidden_states

        # extract spatial query maps
        if self._should_extract_qrys():
            if self.do_spatial_qry_extraction:
                height = int(sqrt(qry.shape[1]))
                qry_square = rearrange(
                    qry,
                    "(b f) (h w) d -> b f d h w",
                    h=height,
                    b=self.chunk_size,
                )
                self.spatial_qry_features[self._cur_module_path] = qry_square

        # extract spatial post attn features
        if self._should_extract_post_attn():
            if self.do_spatial_post_attn_extraction:
                height = int(sqrt(attn_out.shape[1]))
                attn_out_square = rearrange(
                    attn_out,
                    "(b f) (h w) d -> b f d h w",
                    h=height,
                    b=self.chunk_size,
                )
                self.spatial_post_attn_features[self._cur_module_path] = attn_out_square

        return attn_out

    def _call_injection_mode(
        self, attn: Attention, hidden_states: Tensor, attention_mask: Optional[Tensor]
    ):
        """
        Use injected key/value features and rendered post attention features
        """

        def blend_feature_images(original_features_1D: Tensor, feature_images: Tensor):
            # reshape original to 2D
            height = int(sqrt(original_features_1D.shape[1]))
            original_features_2D = rearrange(
                original_features_1D,
                "(b f) (h w) d -> b f d h w",
                h=height,
                b=self.chunk_size,
            )

            # blend rendered and original features
            blended = blend_features(
                original_features_2D,
                feature_images.to(original_features_2D),
                self.feature_blend_alpha,
                channel_dim=2,
            )

            blended_batched = rearrange(blended, "b f d h w -> (b f) d h w")

            # blended_batched = adain_2D(blended_batched, blended_batched)

            blended = rearrange(
                blended_batched, "(b f) d h w -> b f d h w", b=self.chunk_size
            )

            # reshape back to 2D
            blended_features_1D = rearrange(blended, "b f d h w -> (b f) (h w) d")

            return blended_features_1D

        n_frames = hidden_states.shape[0] // self.chunk_size

        injected_kv_features = self.kv_features.get(self._cur_module_path)

        # if no injected kv features, use self-attn
        if injected_kv_features is None:
            kv_features = hidden_states

        # if passed attend to injected kv/features
        else:
            kv_features = extend_across_frame_dim(injected_kv_features, n_frames)

            # concatenate current hidden states
            if self.attend_to_self_kv:
                kv_features = torch.cat([hidden_states, kv_features], dim=1)

        key = attn.to_k(kv_features)
        val = attn.to_v(kv_features)
        qry = attn.to_q(hidden_states)

        # inject query features
        injected_qrys = self.spatial_qry_features.get(self._cur_module_path)
        if injected_qrys is not None:
            qry = blend_feature_images(qry, injected_qrys)

        # self.write_qkv(qry, key, val)

        attn_out = memory_efficient_attention(attn, key, qry, val, attention_mask)

        # inject post attn features
        injected_attn_out = self.spatial_post_attn_features.get(self._cur_module_path)
        if injected_attn_out is not None:
            attn_out = blend_feature_images(attn_out, injected_attn_out)

        return attn_out

    def _call_cross_attn(
        self, attn, hidden_states, encoder_hidden_states, attention_mask
    ):
        qry = attn.to_q(hidden_states)

        kv_hidden_states = encoder_hidden_states
        if attn.norm_cross is not None:
            kv_hidden_states = attn.norm_cross(hidden_states)

        key = attn.to_k(kv_hidden_states)
        val = attn.to_v(kv_hidden_states)

        if self.mode == AttnMode.FEATURE_INJECTION:
            self.write_qkv(qry, key, val)

        return memory_efficient_attention(attn, key, qry, val, attention_mask)

    def _call_self_attn(self, attn, hidden_states, attention_mask):
        if self.mode == AttnMode.FEATURE_EXTRACTION:
            attn_out = self._call_extraction_mode(attn, hidden_states, attention_mask)
        elif self.mode == AttnMode.FEATURE_INJECTION:
            attn_out = self._call_injection_mode(attn, hidden_states, attention_mask)

        return attn_out
