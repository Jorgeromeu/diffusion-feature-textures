from typing import List

import torch
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from text3d2video.attn_processors.style_aligned_attn import (
    StyleAlignedAttentionProcessor,
)
from text3d2video.pipelines.base_pipeline import BaseStableDiffusionPipeline
from text3d2video.utilities.diffusion_data import (
    AttnFeaturesWriter,
    DiffusionDataLogger,
    LatentsWriter,
)


class StyleAlignedLogger(DiffusionDataLogger):
    def __init__(
        self,
        h5_file_path,
        enabled=True,
        path_greenlist=None,
        frame_indices_greenlist=None,
        noise_level_greenlist=None,
    ):
        super().__init__(
            h5_file_path,
            enabled,
            path_greenlist,
            frame_indices_greenlist,
            noise_level_greenlist,
        )

        self.latents_writer = LatentsWriter(self)
        self.attn_writer = AttnFeaturesWriter(self)

    def setup_logger(self, scheduler: SchedulerMixin, n_frames: int):
        self.calc_evenly_spaced_frame_indices(n_frames, -1)
        self.calc_evenly_spaced_noise_noise_levels(scheduler, 8)


class StyleAlignedPipeline(BaseStableDiffusionPipeline):
    attn_processor: StyleAlignedAttentionProcessor
    logger: StyleAlignedLogger

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: UniPCMultistepScheduler,
    ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)

        self.logger = StyleAlignedLogger(None, False)

    @torch.no_grad()
    def __call__(
        self,
        prompts: List[str],
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=None,
        logger=None,
    ):
        # number of images being generated
        batch_size = len(prompts)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # setup logger
        if logger is not None:
            self.logger = logger
            self.logger.setup_logger(self.scheduler, batch_size)

        # Get prompt embeddings for guidance
        cond_embeddings, uncond_embeddings = self.encode_prompt(prompts)
        text_embeddings = torch.cat([cond_embeddings, uncond_embeddings])

        # initialize latents from standard normal
        latents = self.prepare_latents(batch_size, generator)

        # denoising loop
        for _, t in enumerate(tqdm(self.scheduler.timesteps)):
            self.logger.latents_writer.write_latents_batched(t, latents)

            # duplicate latent, to feed to model with CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # diffusion step
            self.attn_processor.cur_timestep = t
            self.attn_processor.chunk_frame_indices = torch.arange(batch_size)

            # give data to attention processor for logging
            self.attn_processor.set_attn_data_writer(self.logger.attn_writer)
            self.attn_processor.set_chunk_frame_indices(torch.arange(batch_size))

            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=text_embeddings,
            ).sample

            # preform classifier free guidance
            noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
            guidance_direction = noise_pred_cond - noise_pred_uncond
            noise_pred = noise_pred_uncond + guidance_scale * guidance_direction

            # update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        self.logger.latents_writer.write_latents_batched(0, latents)

        # decode latents
        return self.decode_latents(latents, generator)
