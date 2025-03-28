from typing import List

import torch
from attr import dataclass
from einops import rearrange
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from text3d2video.attn_processors.extraction_injection_attn import (
    ExtractionInjectionAttn,
)
from text3d2video.backprojection import diffusion_dict_map
from text3d2video.noise_initialization import (
    FixedNoiseInitializer,
    RandomNoiseInitializer,
)
from text3d2video.pipelines.controlnet_pipeline import BaseControlNetPipeline
from text3d2video.utilities.image_utils import Affine2D


@dataclass
class ReposableDiffusion2DConfig:
    # injection settings
    feature_blend_alpha: float
    # UNet Injection settings
    do_kv_injection: bool
    do_qry_injection: bool
    do_post_attn_injection: bool
    spatial_extraction_layers: List[str]
    kv_extraction_layers: List[str]
    # ControlNet Injection Settings
    do_kv_injection_controlnet: bool
    do_qry_injection_controlnet: bool
    do_post_attn_injection_controlnet: bool
    controlnet_layers: List[str]
    # diffusion settings
    guidance_scale: float
    num_inference_steps: int
    controlnet_conditioning_scale: float


class ReposableDiffusion2DPipeline(BaseControlNetPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        src_depth: Image.Image,
        warp_funs: List[Affine2D],
        config: ReposableDiffusion2DConfig,
        generator=None,
    ):
        tgt_depths = [f(src_depth) for f in warp_funs]
        batch_size = 1 + len(warp_funs)

        src_indices = Tensor([0]).long()
        tgt_indices = Tensor(list(range(1, batch_size))).long()

        # Get prompt embeddings for guidance
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt] * batch_size)
        stacked_embeddings = torch.stack([cond_embeddings, uncond_embeddings])

        src_embeddings = stacked_embeddings[:, src_indices, ...]
        tgt_embeddings = stacked_embeddings[:, tgt_indices, ...]

        # set timesteps
        self.scheduler.set_timesteps(config.num_inference_steps)

        # initialize latents from standard normal
        noise_initializer = FixedNoiseInitializer()
        latents = noise_initializer.initial_noise(
            batch_size, device=self.device, dtype=self.dtype, generator=generator
        )

        unet_attn_processor = ExtractionInjectionAttn(
            self.unet,
            do_spatial_qry_extraction=config.do_qry_injection,
            do_spatial_post_attn_extraction=config.do_post_attn_injection,
            do_kv_extraction=config.do_kv_injection,
            also_attend_to_self=False,
            feature_blend_alpha=config.feature_blend_alpha,
            kv_extraction_paths=config.kv_extraction_layers,
            spatial_qry_extraction_paths=config.spatial_extraction_layers,
            spatial_post_attn_extraction_paths=config.spatial_extraction_layers,
        )

        self.unet.set_attn_processor(unet_attn_processor)

        # denoising loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            # duplicate latent, to feed to model with CFG
            latents = self.scheduler.scale_model_input(latents, t)

            # duplicate for CFG
            latents_stacked = torch.stack([latents] * 2)

            src_latents = latents_stacked[:, src_indices, ...]
            tgt_latents = latents_stacked[:, tgt_indices, ...]

            # Process source latents with extraction
            src_latents_batched = rearrange(src_latents, "b f c h w -> (b f) c h w")
            src_embeddings_batched = rearrange(src_embeddings, "b f t d -> (b f) t d")

            processed_ctrl_image = self.preprocess_controlnet_images([src_depth] * 2)
            down_residuals, mid_residual = self.controlnet(
                src_latents_batched,
                t,
                encoder_hidden_states=src_embeddings_batched,
                controlnet_cond=processed_ctrl_image,
                conditioning_scale=config.controlnet_conditioning_scale,
                guess_mode=False,
                return_dict=False,
            )

            unet_attn_processor.set_extraction_mode()
            noise_preds_src = self.unet(
                src_latents_batched,
                t,
                mid_block_additional_residual=mid_residual,
                down_block_additional_residuals=down_residuals,
                encoder_hidden_states=src_embeddings_batched,
            ).sample
            noise_preds_src = rearrange(
                noise_preds_src, "(b f) c h w -> b f c h w", b=2
            )

            unet_kv_features = unet_attn_processor.kv_features
            unet_qrys = unet_attn_processor.spatial_qry_features
            unet_post_attn = unet_attn_processor.spatial_post_attn_features

            # Warp spatial Features to Target

            def warp_to_tgts(x, _):
                fmap = x[0]
                warped = [f(fmap) for f in warp_funs]
                warped = torch.stack(warped)
                return warped

            warped_unet_qrys = diffusion_dict_map(unet_qrys, warp_to_tgts)
            warped_unet_post_attn = diffusion_dict_map(unet_post_attn, warp_to_tgts)

            # TARGET

            tgt_latents_batched = rearrange(tgt_latents, "b f c h w -> (b f) c h w")
            tgt_embeddings_batched = rearrange(tgt_embeddings, "b f t d -> (b f) t d")

            # Target feature extraction for loss calculation
            processed_ctrl_image = self.preprocess_controlnet_images(tgt_depths * 2)
            down_residuals, mid_residual = self.controlnet(
                tgt_latents_batched,
                t,
                encoder_hidden_states=tgt_embeddings_batched,
                controlnet_cond=processed_ctrl_image,
                conditioning_scale=config.controlnet_conditioning_scale,
                guess_mode=False,
                return_dict=False,
            )

            unet_attn_processor.set_injection_mode(
                pre_attn_features=unet_kv_features,
                qry_features=warped_unet_qrys,
                post_attn_features=warped_unet_post_attn,
            )
            unet_attn_processor.set_frame_indices(tgt_indices)
            noise_preds_tgt = self.unet(
                tgt_latents_batched,
                t,
                mid_block_additional_residual=mid_residual,
                down_block_additional_residuals=down_residuals,
                encoder_hidden_states=tgt_embeddings_batched,
            ).sample
            noise_preds_tgt = rearrange(
                noise_preds_tgt, "(b f) c h w -> b f c h w", b=2
            )

            # noise_preds_tgt_no_injection = self.unet(
            #     tgt_latents_batched,
            #     t,
            #     mid_block_additional_residual=mid_residual,
            #     down_block_additional_residuals=down_residuals,
            #     encoder_hidden_states=tgt_embeddings_batched,
            # ).sample
            # noise_preds_tgt_no_injection = rearrange(
            #     noise_preds_tgt, "(b f) c h w -> b f c h w", b=2
            # )

            noise_preds = torch.cat([noise_preds_src, noise_preds_tgt], dim=1)

            # perform classifier-free guidance
            noise_pred_cond, noise_pred_uncond = noise_preds
            guidance_dir = noise_pred_cond - noise_pred_uncond
            noise_pred = noise_pred_uncond + config.guidance_scale * guidance_dir

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents
        return self.decode_latents(latents)
