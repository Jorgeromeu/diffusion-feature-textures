from dataclasses import dataclass

import torch
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from PIL import Image
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


@dataclass
class FeaturePipelineOutput:
    images: list

class FeaturePipeline(DiffusionPipeline):

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: UniPCMultistepScheduler,
    ):

        super().__init__()

        # Register SD modules
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder
        )

    def _init_latents(self, batch_size, res=512):
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, res // 8, res // 8),
            generator=self.generator,
            device=self.device
        )

        return latents

    def _decode_latents(self, latents):
        latents_p = 1 / 0.18215 * latents
        with torch.no_grad():
            images = self.vae.decode(latents_p).sample
        return images

    def _to_pil_images(self, images):
        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        pil_images = [Image.fromarray(image) for image in images]
        return pil_images

    @torch.no_grad()
    def __call__(
        self,
        prompts: list[str],
        res=512,
        num_steps=25,
        guidance_scale=7.5,
        generator=None,
        feature_timestep=3,
        feature_level=2
    ):

        if generator is None:
            self.generator = torch.Generator(device=self.device)
        else:
            self.generator = generator

        batch_size = len(prompts)

        self.scheduler.set_timesteps(num_steps)

        text_input = self.tokenizer(
            prompts, padding='max_length', max_length=self.tokenizer.model_max_length, truncation=True, return_tensors='pt'
        )

        # Get CLIP embedding
        with torch.no_grad():
            text_embeddings = self.text_encoder(
                text_input.input_ids.to(self.device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings = self.text_encoder(
            uncond_input.input_ids.to(self.device))[0]

        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        # initialize latents from standard normal
        latents = self._init_latents(batch_size, res)

        # diffusion process
        for t in tqdm(self.scheduler.timesteps):

            self.cur_timestep = t

            # duplicate latent
            latent_model_input = torch.cat([latents]*2)

            # diffusion step
            with torch.no_grad():

                noise_pred = self.unet(
                    latent_model_input, t, encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * \
                (noise_pred_text - noise_pred_uncond)

            # perform step
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents
        images = self._decode_latents(latents)
        pil_images = self._to_pil_images(images)

        return FeaturePipelineOutput(images=pil_images)
