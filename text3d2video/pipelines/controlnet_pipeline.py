from typing import List, Optional

import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from jaxtyping import Float
from PIL import Image
from torch import Tensor
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from text3d2video.pipelines.base_pipeline import BaseStableDiffusionPipeline


class BaseControlNetPipeline(BaseStableDiffusionPipeline):
    """
    Base Class for Stable Diffusion + ControlNet Pipelines
    """

    def get_partial_timesteps(self, num_inference_steps: int, noise_level: float):
        self.scheduler.set_timesteps(num_inference_steps)

        start_index = int((len(self.scheduler.timesteps) - 1) * noise_level)
        timesteps = self.scheduler.timesteps[start_index:]
        return timesteps

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

    def model_forward(
        self,
        latents: Float[Tensor, "b c h w"],
        embeddings: Float[Tensor, "b t d"],
        t: int,
        depth_maps: Optional[List[Image.Image]],
        controlnet_conditioning_scale=1,
    ):
        if depth_maps is None:
            return self.unet(latents, t, encoder_hidden_states=embeddings).sample

        # ControlNet Pass
        processed_images = self.preprocess_controlnet_images(depth_maps)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latents,
            t,
            encoder_hidden_states=embeddings,
            controlnet_cond=processed_images,
            conditioning_scale=controlnet_conditioning_scale,
            guess_mode=False,
            return_dict=False,
        )

        # UNet Pass
        noise_pred = self.unet(
            latents,
            t,
            encoder_hidden_states=embeddings,
            mid_block_additional_residual=mid_block_res_sample,
            down_block_additional_residuals=down_block_res_samples,
        ).sample

        return noise_pred

    def model_forward_cfg(
        self,
        latents: Float[Tensor, "b c h w"],
        cond_embeddings: Float[Tensor, "b t d"],
        uncond_embeddings: Float[Tensor, "b d"],
        t: int,
        depth_maps: List[Image.Image],
        controlnet_conditioning_scale: float = 1.0,
        guidance_scale: float = 7.5,
    ) -> Tensor:
        """
        Forward pass through ControlNet and UNet
        """

        latents_duplicated = torch.cat([latents] * 2)
        both_embeddings = torch.cat([cond_embeddings, uncond_embeddings])

        noise_pred = self.model_forward(
            latents_duplicated,
            both_embeddings,
            t,
            depth_maps,
            controlnet_conditioning_scale,
        )

        noise_cond, noise_uncond = noise_pred.chunk(2)

        noise_pred = noise_uncond + guidance_scale * (noise_cond - noise_uncond)
        return noise_pred

    @torch.no_grad()
    def __call__(
        self,
        prompts: List[str],
        depth_maps: List[Image.Image],
        num_inference_steps=30,
        controlnet_conditioning_scale=1.0,
        guidance_scale=7.5,
        latents=None,
        t_start=None,
        generator=None,
    ):
        # number of images being generated
        batch_size = len(prompts)

        # encode prompt
        cond_embeddings, uncond_embeddings = self.encode_prompt(prompts)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        if t_start is None:
            t_start = self.scheduler.timesteps[0]

        start_index = (self.scheduler.timesteps >= t_start).nonzero().max()
        denoising_timesteps = self.scheduler.timesteps[start_index:]

        # initialize latents from standard normal
        if latents is None:
            latents = self.prepare_latents(batch_size, generator=generator)

        # denoising loop
        for t in tqdm(denoising_timesteps):
            latents = self.scheduler.scale_model_input(latents, t)

            noise_pred = self.model_forward_cfg(
                latents,
                cond_embeddings,
                uncond_embeddings,
                t,
                depth_maps,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
                guidance_scale=guidance_scale,
            )

            # update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents
        return self.decode_latents(latents)
