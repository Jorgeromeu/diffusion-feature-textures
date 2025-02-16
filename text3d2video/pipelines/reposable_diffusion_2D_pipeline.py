from typing import List

import torch
from attr import dataclass
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DiffusionPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from einops import rearrange
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from text3d2video.backprojection import diffusion_dict_map
from text3d2video.generative_rendering.extraction_injection_attn import (
    ExtractionInjectionAttn,
)
from text3d2video.noise_initialization import (
    FixedNoiseInitializer,
    RandomNoiseInitializer,
)
from text3d2video.utilities.image_utils import Affine2D


@dataclass
class ReposableDiffusion2DConfig:
    do_kv_injection: bool
    do_qry_injection: bool
    do_post_attn_injection: bool
    feature_blend_alpha: float
    unet_attn_layers: List[str]
    # diffusion settings
    guidance_scale: float
    num_inference_steps: int
    controlnet_conditioning_scale: float


class ReposableDiffusion2DPipeline(DiffusionPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: UniPCMultistepScheduler,
        controlnet: ControlNetModel,
    ):
        super().__init__()

        # register modules
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            controlnet=controlnet,
        )

        # vae image processors
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

    def encode_prompt(self, prompts: List[str]):
        # tokenize prompts
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Get CLIP embedding
        with torch.no_grad():
            cond_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * len(prompts),
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        return cond_embeddings, uncond_embeddings

    def prepare_latents(self, batch_size: int, out_resolution: int, generator=None):
        latent_res = out_resolution // 8
        in_channels = self.unet.config.in_channels

        latents = torch.randn(
            batch_size,
            in_channels,
            latent_res,
            latent_res,
            device=self.device,
            generator=generator,
            dtype=self.dtype,
        )

        return latents

    def latents_to_images(self, latents: torch.FloatTensor, generator=None):
        # scale latents
        latents_scaled = latents / self.vae.config.scaling_factor

        # decode latents
        images = self.vae.decode(
            latents_scaled,
            return_dict=False,
            generator=generator,
        )[0]

        # postprocess images
        images = self.image_processor.postprocess(
            images, output_type="pil", do_denormalize=[True] * len(latents)
        )

        return images

    def prepare_controlnet_image(
        self, images: List[Image.Image], do_classifier_free_guidance=True
    ):
        height = images[0].height
        width = images[0].width

        image = self.control_image_processor.preprocess(
            images, height=height, width=width
        ).to(dtype=self.dtype, device=self.device)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    def preprocess_controlnet_images(self, images: List[Image.Image]):
        height = images[0].height
        width = images[0].width

        image = self.control_image_processor.preprocess(
            images, height=height, width=width
        ).to(dtype=self.dtype, device=self.device)

        return image

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        src_depth: Image.Image,
        warp_funs: List[Affine2D],
        config: ReposableDiffusion2DConfig,
        res=512,
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
        noise_initializer = RandomNoiseInitializer()
        latents = noise_initializer.initial_noise(
            batch_size, device=self.device, dtype=self.dtype, generator=generator
        )

        # latents = self.prepare_latents(batch_size, res, generator=generator)

        controlnet_sa_paths = [
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1",
            "down_blocks.0.attentions.1.transformer_blocks.0.attn1",
            "down_blocks.1.attentions.0.transformer_blocks.0.attn1",
            "down_blocks.1.attentions.1.transformer_blocks.0.attn1",
            "down_blocks.2.attentions.0.transformer_blocks.0.attn1",
            "down_blocks.2.attentions.1.transformer_blocks.0.attn1",
        ]

        unet_attn_processor = ExtractionInjectionAttn(
            self.unet,
            do_spatial_qry_extraction=config.do_qry_injection,
            do_spatial_post_attn_extraction=config.do_post_attn_injection,
            do_kv_extraction=config.do_kv_injection,
            attend_to_self_kv=False,
            feature_blend_alpha=config.feature_blend_alpha,
            extraction_attn_paths=config.unet_attn_layers,
        )

        controlnet_attn_processor = ExtractionInjectionAttn(
            self.controlnet,
            do_spatial_qry_extraction=True,
            do_spatial_post_attn_extraction=True,
            do_kv_extraction=True,
            attend_to_self_kv=False,
            feature_blend_alpha=1.0,
            extraction_attn_paths=controlnet_sa_paths,
        )

        self.unet.set_attn_processor(unet_attn_processor)
        self.controlnet.set_attn_processor(controlnet_attn_processor)

        # denoising loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            # duplicate latent, to feed to model with CFG
            latents = self.scheduler.scale_model_input(latents, t)

            # duplicate for CFG
            latents_stacked = torch.stack([latents] * 2)

            src_latents = latents_stacked[:, src_indices, ...]
            tgt_latents = latents_stacked[:, tgt_indices, ...]

            # Source with Extraction

            src_latents_batched = rearrange(src_latents, "b f c h w -> (b f) c h w")
            src_embeddings_batched = rearrange(src_embeddings, "b f t d -> (b f) t d")

            controlnet_attn_processor.set_extraction_mode()
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

            controlnet_kv_features = controlnet_attn_processor.kv_features
            controlnet_qrys = controlnet_attn_processor.spatial_qry_features

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

            warped_controlnet_qrys = diffusion_dict_map(controlnet_qrys, warp_to_tgts)
            warped_unet_qrys = diffusion_dict_map(unet_qrys, warp_to_tgts)
            warped_unet_post_attn = diffusion_dict_map(unet_post_attn, warp_to_tgts)

            # Target with Injection

            tgt_latents_batched = rearrange(tgt_latents, "b f c h w -> (b f) c h w")
            tgt_embeddings_batched = rearrange(tgt_embeddings, "b f t d -> (b f) t d")

            # controlnet_attn_processor.set_injection_mode(
            #     pre_attn_features=controlnet_kv_features,
            #     qry_features=warped_controlnet_qrys,
            # )

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
            unet_attn_processor.set_chunk_frame_indices(tgt_indices)
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

            noise_preds = torch.cat([noise_preds_src, noise_preds_tgt], dim=1)

            # perform classifier-free guidance
            noise_pred_cond, noise_pred_uncond = noise_preds
            guidance_dir = noise_pred_cond - noise_pred_uncond
            noise_pred = noise_pred_uncond + config.guidance_scale * guidance_dir

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents
        return self.latents_to_images(latents)
