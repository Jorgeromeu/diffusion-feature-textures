from typing import List

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from text3d2video.pipelines.base_pipeline import BaseStableDiffusionPipeline


class BaseControlNetPipeline(BaseStableDiffusionPipeline):
    """
    Base Class for Stable Diffusion + ControlNet Pipelines
    """

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: UniPCMultistepScheduler,
        controlnet: ControlNetModel,
    ):
        super().__init__(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet,
            scheduler=scheduler,
        )

        # register controlnet
        self.register_modules(
            controlnet=controlnet,
        )

        # controlnet image processor
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

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
        prompts: List[str],
        depth_maps: List[Image.Image],
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        guess_mode=False,
        generator=None,
    ):
        # number of images being generated
        batch_size = len(prompts)

        # encode prompt
        cond_embeddings, uncond_embeddings = self.encode_prompt(prompts)
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # initialize latents from standard normal
        latents = self.prepare_latents(batch_size, generator=generator)

        # denoising loop
        for t in tqdm(self.scheduler.timesteps):
            # duplicate latent for guidance
            latents = self.scheduler.scale_model_input(latents, t)
            latents_duplicated = torch.cat([latents] * 2)

            if guess_mode:
                # Do ControlNet pass on cond latents only
                processed_ctrl_images = self.preprocess_controlnet_images(depth_maps)
                down_residuals, mid_residual = self.controlnet(
                    latents,
                    t,
                    encoder_hidden_states=cond_embeddings,
                    controlnet_cond=processed_ctrl_images,
                    conditioning_scale=controlnet_conditioning_scale,
                    guess_mode=False,
                    return_dict=False,
                )

                mid_residual = torch.cat([torch.zeros_like(mid_residual), mid_residual])
                down_residuals = [
                    torch.cat([torch.zeros_like(residual), residual])
                    for residual in down_residuals
                ]

            # controlnet step
            else:
                # Do ControlNet pass on cond and uncond latents
                processed_ctrl_images = self.preprocess_controlnet_images(depth_maps)
                processed_ctrl_images = torch.cat([processed_ctrl_images] * 2)
                down_residuals, mid_residual = self.controlnet(
                    latents_duplicated,
                    t,
                    encoder_hidden_states=text_embeddings,
                    controlnet_cond=processed_ctrl_images,
                    conditioning_scale=controlnet_conditioning_scale,
                    guess_mode=False,
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
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents
        return self.decode_latents(latents)
