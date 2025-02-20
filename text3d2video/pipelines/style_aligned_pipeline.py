from typing import List

import torch
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from text3d2video.attn_processors.style_aligned_attn import (
    StyleAlignedAttentionProcessor,
)
from text3d2video.pipelines.base_pipeline import BaseStableDiffusionPipeline


class StyleAlignedPipeline(BaseStableDiffusionPipeline):
    attn_processor: StyleAlignedAttentionProcessor

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: UniPCMultistepScheduler,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)

    @torch.no_grad()
    def __call__(
        self,
        prompts: List[str],
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=None,
    ):
        # number of images being generated
        batch_size = len(prompts)

        # Get prompt embeddings for guidance
        cond_embeddings, uncond_embeddings = self.encode_prompt(prompts)
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # initialize latents from standard normal
        latents = self.prepare_latents(batch_size, generator)

        # denoising loop
        for _, t in enumerate(tqdm(self.scheduler.timesteps)):
            self.data_artifact.latents_writer.write_latents_batched(t, latents)

            # duplicate latent, to feed to model with CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # diffusion step
            self.attn_processor.cur_timestep = t
            self.attn_processor.chunk_frame_indices = torch.arange(batch_size)
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample

            # preform classifier free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guidance_direction = noise_pred_text - noise_pred_uncond
            noise_pred = noise_pred_uncond + guidance_scale * guidance_direction

            # update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        self.data_artifact.latents_writer.write_latents_batched(0, latents)

        # decode latents
        return self.decode_latents(latents, generator)
