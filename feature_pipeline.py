from typing import Dict, List, Set, Tuple
import torch
from PIL import Image
from diffusers import (
    DiffusionPipeline, AutoencoderKL, UNet2DConditionModel,
    UniPCMultistepScheduler
)

from transformers import CLIPTokenizer, CLIPTextModel
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class FeaturePipelineOutput:
    images: list


class FeaturePipeline(DiffusionPipeline):

    # for each timestep and level store feature
    sd_features: Dict[Tuple[int, int], torch.Tensor] = dict()
    # keep track of timesteps and levels we save features at
    feature_levels: Set[int] = set()
    feature_timesteps: Set[int] = set()

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

        # hooks that save the features of the up convs
        self._setup_hooks()

    def _save_feature_hook(self, level):

        def hook(module, inp, out):
            self.feature_timesteps.add(self.cur_timestep.item())
            # save the feature
            self.sd_features[(self.cur_timestep.item(), level)
                             ] = out.cpu().numpy()

        return hook

    def _setup_hooks(self):
        """
        Set up UNet hooks to extract Up Conv features at each timestep
        """

        unet: UNet2DConditionModel = self.unet

        # for each level register a hook that saves the output of the up conv
        for level in range(len(unet.up_blocks)):
            self.feature_levels.add(level)
            unet.up_blocks[level].register_forward_hook(
                self._save_feature_hook(level))

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

        # eval config
        self.feature_level = feature_level
        self.feature_timestep = feature_timestep

        self.sd_features = dict()

        if generator is None:
            self.generator = torch.Generator(device=self.device)
            self.seed()
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
