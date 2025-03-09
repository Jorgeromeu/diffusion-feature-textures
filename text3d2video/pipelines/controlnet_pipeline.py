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

    def controlnet_forward(self, images, latents, t, embeddings, conditioning_scale):
        processed_images = self.preprocess_controlnet_images(images)

        down_residuals, mid_residual = self.controlnet(
            latents,
            t,
            encoder_hidden_states=embeddings,
            controlnet_cond=processed_images,
            conditioning_scale=conditioning_scale,
            guess_mode=False,
            return_dict=False,
        )

        return down_residuals, mid_residual

    @torch.no_grad()
    def __call__(
        self,
        prompts: List[str],
        depth_maps: List[Image.Image],
        num_inference_steps=30,
        controlnet_conditioning_scale=1.0,
        w_joint=1.0,
        w_text=0.0,
        generator=None,
    ):
        # number of images being generated
        batch_size = len(prompts)

        # encode prompt
        cond_embeddings, uncond_embeddings = self.encode_prompt(prompts)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # initialize latents from standard normal
        latents = self.prepare_latents(batch_size, generator=generator)

        # denoising loop
        for t in tqdm(self.scheduler.timesteps):
            latents = self.scheduler.scale_model_input(latents, t)

            # Do 4 passes through UNet:
            # 0: no condition
            # 1: text only
            # 2: depth only
            # 3: text and depth

            # ControlNet pass with cond and uncond embeddings
            down_residuals, mid_residual = self.controlnet_forward(
                depth_maps * 2,
                torch.cat([latents] * 2),
                t,
                torch.cat([uncond_embeddings, cond_embeddings]),
                controlnet_conditioning_scale,
            )

            empty_mid_residual = torch.zeros_like(mid_residual[0:batch_size, ...])
            empty_down_residuals = [
                torch.zeros_like(residual[0:batch_size, ...])
                for residual in down_residuals
            ]

            mid_residual = torch.cat(
                [empty_mid_residual, empty_mid_residual, mid_residual], dim=0
            )
            down_residuals = [
                torch.cat([empty, empty, residual], dim=0)
                for empty, residual in zip(empty_down_residuals, down_residuals)
            ]

            # UNet Pass
            latents_input = torch.cat([latents] * 4, dim=0)
            embeddings_input = torch.cat(
                [
                    uncond_embeddings,
                    cond_embeddings,
                    uncond_embeddings,
                    cond_embeddings,
                ],
            )

            noise_pred_all = self.unet(
                latents_input,
                t,
                mid_block_additional_residual=mid_residual,
                down_block_additional_residuals=down_residuals,
                encoder_hidden_states=embeddings_input,
            ).sample

            # preform classifier free guidance
            eps_uncond, eps_text, eps_depth, eps_text_depth = noise_pred_all.chunk(4)

            # eps = eps_depth + 7.5 * (eps_text_depth - eps_depth)

            eps = (
                eps_uncond
                + 7.5 * (eps_text - eps_uncond)
                + 4.5 * (eps_text_depth - eps_uncond)
            )

            # update latents
            latents = self.scheduler.step(eps, t, latents).prev_sample

        # decode latents
        return self.decode_latents(latents)
