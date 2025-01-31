from typing import List

import torch
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from text3d2video.artifacts.sd_data import SdDataArtifact, SdDataConfig
from text3d2video.noise_initialization import (
    FixedNoiseInitializer,
    RandomNoiseInitializer,
)
from text3d2video.style_aligned_attn import StyleAlignedAttentionProcessor


class SDPipeline(DiffusionPipeline):
    attn_processor: StyleAlignedAttentionProcessor
    data_artifact: SdDataArtifact

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: UniPCMultistepScheduler,
    ):
        super().__init__()

        # register modules
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
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
        noise_init = RandomNoiseInitializer()
        noise_init = FixedNoiseInitializer()
        return noise_init.initial_noise(
            generator=generator,
            device=self.device,
            dtype=self.dtype,
            n_frames=batch_size,
        )

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

    @torch.no_grad()
    def __call__(
        self,
        prompts: List[str],
        sd_save_config: SdDataConfig,
        res=512,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=None,
    ):
        self.data_artifact = SdDataArtifact.init_from_config(sd_save_config)
        self.data_artifact.begin_recording(self.scheduler, len(prompts))
        self.attn_processor.attn_writer = self.data_artifact.attn_writer

        # number of images being generated
        batch_size = len(prompts)

        # Get prompt embeddings for guidance
        cond_embeddings, uncond_embeddings = self.encode_prompt(prompts)
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # initialize latents from standard normal
        latents = self.prepare_latents(batch_size, res, generator)

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
        return self.latents_to_images(latents, generator)
