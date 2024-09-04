from typing import List
import torch
from PIL import Image
from diffusers import (
    DiffusionPipeline, AutoencoderKL, UNet2DConditionModel,
    UniPCMultistepScheduler, ControlNetModel
)

from tqdm import tqdm
from transformers import CLIPTokenizer, CLIPTextModel
from dataclasses import dataclass

class FeatureControlNetPipeline(DiffusionPipeline):

        
    def __init__(
        self, 
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: UniPCMultistepScheduler,
        controlnet: ControlNetModel
    ):
        
        super().__init__()
        
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            controlnet=controlnet
        )

    def _init_latents(self, batch_size, res=512):
        latents = torch.randn(
            (batch_size, self.unet.config.in_channels, res // 8, res // 8),
            generator=self.generator,
            device=self.device
        ) 
        
        return latents

    def __call__(
            self,
            prompt: List[str],
            res=512,
            num_inference_steps=25
    ):
        
        batch_size = len(prompt)
        
        