from dataclasses import dataclass
from typing import List

import torch
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from text3d2video.artifacts.diffusion_data import (
    AttnFeaturesWriter,
    DiffusionDataManager,
    LatentsWriter,
)
from text3d2video.attn_processors.style_aligned_attn import (
    StyleAlignedAttentionProcessor,
)
from text3d2video.pipelines.base_pipeline import BaseStableDiffusionPipeline


@dataclass
class StyleAlignedDataWriterConfig:
    enabled: bool
    save_latents: bool
    n_noise_levels: int
    attn_layers: List[str]


class StyleAlignedDataWriter(DiffusionDataManager):
    def __init__(self, h5_file_path, save_config: StyleAlignedDataWriterConfig):
        super().__init__(h5_file_path, enabled=save_config.enabled)
        self.layer_paths = save_config.attn_layers
        self.latents_writer = LatentsWriter(
            self, data_path="latents", enabled=save_config.save_latents
        )
        self.attn_writer = AttnFeaturesWriter(
            self,
            data_path="attn",
            save_q=True,
            save_k=True,
            save_v=True,
        )

    @classmethod
    def create_disabled(cls):
        return cls(
            None,
            StyleAlignedDataWriterConfig(
                enabled=False, save_latents=False, n_noise_levels=0, attn_layers=[]
            ),
        )


class StyleAlignedPipeline(BaseStableDiffusionPipeline):
    attn_processor: StyleAlignedAttentionProcessor
    writer: StyleAlignedDataWriter

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
        save_data_file=None,
        save_data_config: StyleAlignedDataWriterConfig = None,
        generator=None,
    ):
        # number of images being generated
        batch_size = len(prompts)

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # setup recording
        if save_data_file is not None:
            self.writer = StyleAlignedDataWriter(save_data_file, save_data_config)
            self.writer.calculate_evenly_spaced_save_frames(batch_size, -1)
            self.writer.calculate_evenly_spaced_save_levels(
                self.scheduler, save_data_config.n_noise_levels
            )
            self.writer.begin_recording()
        else:
            self.writer = StyleAlignedDataWriter.create_disabled()

        # Get prompt embeddings for guidance
        cond_embeddings, uncond_embeddings = self.encode_prompt(prompts)
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        # initialize latents from standard normal
        latents = self.prepare_latents(batch_size, generator)

        # denoising loop
        for _, t in enumerate(tqdm(self.scheduler.timesteps)):
            self.writer.latents_writer.write_latents_batched(t, latents)

            # duplicate latent, to feed to model with CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # diffusion step
            self.attn_processor.cur_timestep = t
            self.attn_processor.chunk_frame_indices = torch.arange(batch_size)
            self.attn_processor.set_attn_data_writer(self.writer.attn_writer)

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

        self.writer.latents_writer.write_latents_batched(0, latents)

        self.writer.end_recording()

        # decode latents
        return self.decode_latents(latents, generator)
