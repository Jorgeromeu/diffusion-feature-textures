from typing import Optional
import torch
from einops import einsum, rearrange
from diffusers.models.attention_processor import Attention
from math import sqrt

from text3d2video.sd_feature_extraction import get_module_path


class CrossFrameAttnProcessor:

    feature_images: torch.Tensor
    module_path: str

    def __init__(self, unet, unet_chunk_size=2, do_cross_frame_attn=True):
        """
        :param unet_chunk_size:
            number of batches for each generated image, 2 for classifier free guidance
        """
        self.unet = unet
        self.unet_chunk_size = unet_chunk_size
        self.do_cross_frame_attn = do_cross_frame_attn

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
    ):

        # hidden_states: (batch_size, sequence_length, c)
        batch_size, sequence_length, _ = hidden_states.shape

        # number of frames
        video_length = batch_size // self.unet_chunk_size
        # feature map size (assume square image)
        feature_map_size = int(sqrt(sequence_length))

        attention_mask = attn.prepare_attention_mask(
            attention_mask, sequence_length, batch_size
        )

        # project to query
        query = attn.to_q(hidden_states)

        # Sparse Attention
        is_cross_attention = encoder_hidden_states is not None

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross is not None:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        # project to key, value
        # (batch_size, sequence_length, c)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if not is_cross_attention:

            # number of frames
            video_length = key.size()[0] // self.unet_chunk_size
            former_frame_index = [0] * video_length

            # duplicate first frame for all frame
            key = rearrange(key, "(b f) t c -> b f t c", f=video_length)
            key = key[:, former_frame_index].detach().clone()
            key = rearrange(key, "b f t c -> (b f) t c")

            # duplicate first frame for all frame
            value = rearrange(value, "(b f) t c -> b f t c", f=video_length)
            value = value[:, former_frame_index].detach().clone()
            value = rearrange(value, "b f t c -> (b f) t c")

        # reshape from (batch_size, seq_len, dim) to (batch_size, seq_len, heads, dim//heads)
        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # compute attention scores
        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        # apply attention
        hidden_states = einsum(attention_probs, value, "b t1 t2, b t2 c -> b t1 c")
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj to output dim
        hidden_states = attn.to_out[0](hidden_states)

        # reshape to square
        hidden_states_square = rearrange(
            hidden_states,
            "(b f) (h w) c -> b f h w c",
            f=video_length,
            h=feature_map_size,
        )

        path = get_module_path(self.unet, attn)

        if path == self.module_path:

            feature_images_reshaped = rearrange(
                self.feature_images, "f c h w -> f h w c"
            )
            feature_images_expended = torch.stack(
                [feature_images_reshaped] * self.unet_chunk_size, dim=0
            ).to(hidden_states_square.dtype)

            # inject feature
            hidden_states_square = hidden_states_square + feature_images_expended

        # flatten features
        hidden_states = rearrange(hidden_states_square, "b f h w c -> (b f) (h w) c")

        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states
