from itertools import chain
from typing import List

import torch
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from PIL.Image import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from text3d2video.noise_initialization import (
    RandomNoiseInitializer,
)
from text3d2video.util import split_into_chunks


class BaseStableDiffusionPipeline(DiffusionPipeline):
    """
    Simple Base Stable Diffusion Pipelien we can override with custom behavior
    """

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

    def encode_prompt(self, prompts: List[str], negative_prompts=None):
        # tokenize prompts
        cond_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Get CLIP embedding
        with torch.no_grad():
            cond_embs = self.text_encoder(cond_input.input_ids.to(self.device))[0]

        if negative_prompts is not None:
            assert len(prompts) == len(
                negative_prompts
            ), "prompts and negative prompts must be the same length"

            uncond_input = self.tokenizer(
                negative_prompts,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

        else:
            # create uncond embedding
            max_length = cond_input.input_ids.shape[-1]
            uncond_input = self.tokenizer(
                [""] * len(prompts),
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

        with torch.no_grad():
            uncond_embs = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        return cond_embs, uncond_embs

    def prepare_latents(self, batch_size: int, generator=None):
        noise_init = RandomNoiseInitializer()
        return noise_init.initial_noise(
            generator=generator,
            device=self.device,
            dtype=self.dtype,
            n_frames=batch_size,
        )

    def decode_latents(
        self,
        latents: torch.FloatTensor,
        generator=None,
        output_type="pil",
        chunk_size=10,
    ):
        # scale latents
        scaling_factor = self.vae.config.scaling_factor
        latents_scaled = latents / scaling_factor

        decoded_images = []
        for chunk in split_into_chunks(latents_scaled, chunk_size):
            # decode latents
            images = self.vae.decode(
                chunk,
                return_dict=False,
                generator=generator,
            )[0]

            # postprocess images
            images = self.image_processor.postprocess(
                images, output_type=output_type, do_denormalize=[True] * len(chunk)
            )
            decoded_images.append(images)

        images = list(chain.from_iterable(decoded_images))

        return images

    def encode_images(
        self, images: List[Image], generator=None, chunk_size=10
    ) -> torch.FloatTensor:
        images_chunks = split_into_chunks(images, chunk_size)

        encoded_chunks = []
        for chunk in images_chunks:
            # preprocess image
            images_processed = self.image_processor.preprocess(chunk).to(
                device=self.device, dtype=self.dtype
            )

            # encode
            encoded = self.vae.encode(images_processed).latent_dist.sample(
                generator=generator
            )

            # scale latents
            scaling_factor = self.vae.config.scaling_factor
            encoded = encoded * scaling_factor
            encoded_chunks.append(encoded)

        return torch.cat(encoded_chunks, dim=0)

    def get_partial_timesteps(self, num_inference_steps: int, noise_level: float):
        self.scheduler.set_timesteps(num_inference_steps)

        start_index = int((len(self.scheduler.timesteps) - 1) * noise_level)
        timesteps = self.scheduler.timesteps[start_index:]
        return timesteps

    @torch.no_grad()
    def __call__(
        self,
        prompts: List[str],
        num_inference_steps=30,
        guidance_scale=7.5,
        start_latents=None,
        start_noise_level=0,
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
        if start_latents is None:
            latents = self.prepare_latents(batch_size, generator)
        else:
            latents = start_latents.clone()

        timesteps = self.get_partial_timesteps(10, 0.5)
        print(timesteps[0])

        # denoising loop
        for _, t in enumerate(tqdm(timesteps)):
            # duplicate latent, to feed to model with CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # diffusion step
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

        # decode latents
        return self.decode_latents(latents, generator)

    def noise_variance(self, t):
        """
        Return noise scale at timestep t, e.g alpha_t [0, 1]
        """
        return 1 - self.scheduler.alphas_cumprod[t]

    def denoising_progress(self, t):
        """
        Return ratio of denosing steps done, starts at 0, ends at 1
        """
        timesteps = self.scheduler.timesteps
        index = (timesteps == t).nonzero(as_tuple=True)[0].item()
        return index / (len(timesteps) - 1)
