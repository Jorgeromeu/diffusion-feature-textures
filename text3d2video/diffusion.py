import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)
from safetensors.torch import *

from text3d2video.pipelines.my_pipeline import MyPipeline


def make_controlnet_diffusion_pipeline(
    sd_repo: str,
    controlnet_repo: str,
    torch_dtype: torch.dtype = torch.float16,
    device="cuda:0",
) -> MyPipeline:

    device = torch.device("cuda")

    controlnet = ControlNetModel.from_pretrained(
        controlnet_repo, torch_dtype=torch.float16
    ).to(device)

    pipe = MyPipeline.from_pretrained(
        sd_repo, controlnet=controlnet, torch_dtype=torch_dtype
    ).to(device)

    return pipe
