from math import sqrt
from typing import Dict, List, Optional

import rerun as rr
import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention
from einops import rearrange
from jaxtyping import Float

from pytorch3d.structures import Meshes
from pytorch3d.renderer import FoVPerspectiveCameras

from text3d2video.rendering import make_feature_renderer
from text3d2video.util import (
    aggregate_features_precomputed_vertex_positions,
    blend_features,
    project_vertices_to_cameras,
)

from pytorch3d.renderer import TexturesVertex


class IdentityPostAttnInjectionProcessor:

    mesh: Meshes
    camera: FoVPerspectiveCameras

    def __init__(self, unet_chunk_size=2):
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

        # else use ordinary hidden states
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

        query = attn.to_q(hidden_states)

        # get hidden states for k/v computation
        kv_hidden_states = self.get_kv_hidden_states(
            attn, hidden_states, encoder_hidden_states
        )

        key = attn.to_k(kv_hidden_states)
        value = attn.to_v(kv_hidden_states)

        # compute attention
        attn_out = self.memory_efficient_attention(
            attn, key, query, value, attention_mask
        )

        # reshape to square
        feature_map_res = int(sqrt(attn_out.shape[1]))
        attn_out_square = rearrange(
            attn_out,
            "(b f) (h w) d -> b f d h w",
            h=feature_map_res,
            b=self.unet_chunk_size,
        )
        n_frames = attn_out_square.shape[1]

        # prjoect to mesh
        vert_xys, vert_indices = project_vertices_to_cameras(self.mesh, self.camera)

        if self.do_identity_post_attn:

            all_feature_images = []
            for batch_attn_out_square in attn_out_square:

                # project features to mesh
                vert_features = aggregate_features_precomputed_vertex_positions(
                    batch_attn_out_square,
                    self.mesh.num_verts_per_mesh()[0],
                    vert_xys,
                    vert_indices,
                    aggregation_type="first",
                )

                # render features to 2D
                renderer = make_feature_renderer(self.camera, feature_map_res)
                tex = TexturesVertex(vert_features.expand(n_frames, -1, -1))
                self.mesh.textures = tex
                feature_images = renderer(self.mesh)

                all_feature_images.append(feature_images)

            feature_images = torch.stack(all_feature_images, dim=0)
            feature_images = rearrange(feature_images, "b f h w c -> b f c h w")

            # blend
            attn_out_square = blend_features(
                attn_out_square,
                feature_images.to(attn_out),
                1,
                channel_dim=2,
            )

            attn_out_square = rearrange(attn_out_square, "b f c h w -> b f c w h")

        # reshape to flat
        attn_out = rearrange(attn_out_square, "b f d h w -> (b f) (h w) d")

        # linear proj to output dim
        attn_out = attn.to_out[0](attn_out)
        attn_out = attn.to_out[1](attn_out)

        return attn_out
