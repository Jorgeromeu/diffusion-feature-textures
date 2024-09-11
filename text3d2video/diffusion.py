import torch
from safetensors.torch import *
from huggingface_hub import hf_hub_download
from PIL import Image
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler
)

DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"
ckpt = "diffusion_pytorch_model.fp16.safetensors"
repo = "runwayml/stable-diffusion-v1-5"


def depth2img_pipe(device='cuda:0'):
    
    """
    Construct a depth2img pipeline
    """

    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-depth", torch_dtype=torch.float16
    ).to(device)

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        controlnet=controlnet,
        safety_checker=None,
        torch_dtype=torch.float16
    ).to(device)
    
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

    return pipe

def depth2img(
    pipe: StableDiffusionControlNetPipeline,
    prompts: list[str],
    depths: list[Image],
    guidance_scale=7,
    num_inference_steps=30

):
    
    neg_prompt='lowres, low quality, monochrome, watermark',
    pos_promppt_supplement = 'best quality, highly detailed, photorealistic, 3D Render, black background'

    prompts_modified = [f'{prompt}, {pos_promppt_supplement}' for prompt in prompts]
    
    output = pipe(
        prompts_modified,
        neg_prompt=neg_prompt,
        image=depths, 
        guidance_scale=guidance_scale,
        eta=1,
        num_inference_steps=num_inference_steps
    )
    
    return output.images