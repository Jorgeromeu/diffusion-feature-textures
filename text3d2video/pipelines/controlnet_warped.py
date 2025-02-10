from typing import List

import torch
import torchvision.transforms.functional as TF
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DiffusionPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typeguard import typechecked

from text3d2video.utilities.image_utils import Affine2D


class WarpedControlNetPipeline(DiffusionPipeline):
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
        height, width = TF.get_image_size(images[0])
        image = self.control_image_processor.preprocess(
            images, height=height, width=width
        ).to(dtype=self.dtype, device=self.device)
        return image

    @torch.no_grad()
    @typechecked
    def __call__(
        self,
        prompt: str,
        depth_map: Image.Image,
        src_to_tgt: Affine2D,
        res=512,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        generator=None,
        guess_mode=False,
    ):
        # number of images being generated
        batch_size = 1

        # Get prompt embeddings for guidance
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt])
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        tgt_depth_map = src_to_tgt(depth_map)

        # initialize latents from standard normal
        latents_src = self.prepare_latents(batch_size, res, generator=generator)
        latents_tgt = self.prepare_latents(batch_size, res, generator=generator)

        # denoising loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            # duplicate latent, to feed to model with CFG
            latents_duplicated = torch.cat([latents_src] * 2)
            latents_duplicated = self.scheduler.scale_model_input(latents_duplicated, t)

            # controlnet step on source depth map
            processed_control_image = self.prepare_controlnet_image([depth_map])
            down_residuals, mid_residual = self.controlnet(
                latents_duplicated,
                t,
                encoder_hidden_states=text_embeddings,
                controlnet_cond=processed_control_image,
                conditioning_scale=controlnet_conditioning_scale,
                guess_mode=guess_mode,
                return_dict=False,
            )

            # diffusion step, with controlnet residuals
            noise_pred = self.unet(
                latents_duplicated,
                t,
                mid_block_additional_residual=mid_residual,
                down_block_additional_residuals=down_residuals,
                encoder_hidden_states=text_embeddings,
            ).sample

            # preform classifier free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guidance_direction = noise_pred_text - noise_pred_uncond
            noise_pred = noise_pred_uncond + guidance_scale * guidance_direction

            # update latents
            latents_src = self.scheduler.step(noise_pred, t, latents_src).prev_sample

        # decode latents
        return self.latents_to_images(latents_src)
