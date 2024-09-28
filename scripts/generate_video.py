from typing import List

from diffusers import StableDiffusionControlNetPipeline
from PIL.Image import Image


def generate_video(
    pipe: StableDiffusionControlNetPipeline,
    prompt: str,
    depth_maps: List[Image],
    num_inference_steps: int = 50,
) -> List[Image]:

    prompts = [prompt] * len(depth_maps)

    frames = pipe(
        prompts, image=depth_maps, num_inference_steps=num_inference_steps
    ).images

    return frames
