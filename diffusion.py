import torch
from safetensors.torch import *
from huggingface_hub import hf_hub_download
from diffusers import (
    ControlNetModel,
    UNet2DConditionModel,
    StableDiffusionControlNetImg2ImgPipeline,
    DDIMScheduler
)

DIFFUSION_MODEL_ID = "runwayml/stable-diffusion-v1-5"
ckpt = "diffusion_pytorch_model.fp16.safetensors"
repo = "runwayml/stable-diffusion-v1-5"

def process_depth_map(depth):
    """
    Convert from zbuf to depth map
    """

    max_depth = depth.max()
    indices = depth == -1
    depth = max_depth - depth
    depth[indices] = 0
    max_depth = depth.max()
    depth = depth / max_depth
    return depth

def depth_pipe(device):

    # depth controlnet
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth",
        torch_dtype=torch.float16,
    )

    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        DIFFUSION_MODEL_ID,
        controlnet=controlnet
    ).to(device)

    return pipe

def init_pipe(device):

    """
    Initialize a Depth2Img stable-diffusion ControlNet
    """

    # depth control net
    controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11f1p_sd15_depth",
            torch_dtype=torch.float16,
        )

    # SD UNet
    unet = UNet2DConditionModel.from_config(DIFFUSION_MODEL_ID, subfolder="unet").to(device, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(repo_id=repo, subfolder="unet", filename=ckpt)))

    # construct pipeline 
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        DIFFUSION_MODEL_ID,
        unet=unet,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to(device)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    # pipe.enable_xformers_memory_efficient_attention()

    return pipe
