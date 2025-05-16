import torch
from attr import dataclass
from diffusers import ControlNetModel
from diffusers.schedulers import DDIMScheduler


@dataclass
class ModelConfig:
    sd_repo: str = "runwayml/stable-diffusion-v1-5"
    controlnet_repo: str = "lllyasviel/control_v11f1p_sd15_depth"


def load_pipeline(
    pipeline_class: type,
    sd_path="runwayml/stable-diffusion-v1-5",
    controlnet_path="lllyasviel/control_v11f1p_sd15_depth",
    device="cuda",
    dtype=torch.float16,
    scheduler_class=DDIMScheduler,
):
    device = torch.device("cuda")
    dtype = torch.float16

    kwargs = {}

    if controlnet_path is not None:
        controlnet = ControlNetModel.from_pretrained(
            controlnet_path, torch_dtype=dtype
        ).to(device)
        kwargs["controlnet"] = controlnet

    pipe = pipeline_class.from_pretrained(sd_path, torch_dtype=dtype, **kwargs).to(
        device
    )

    pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)

    return pipe
