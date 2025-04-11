import torch
from attr import dataclass
from diffusers import ControlNetModel
from diffusers.schedulers import DDIMScheduler


@dataclass
class ModelConfig:
    sd_repo: str
    controlnet_repo: str


def load_pipeline(
    pipeline_class: type,
    sd_path=None,
    controlnet_path=None,
    device="cuda",
    dtype=torch.float16,
    scheduler_class=DDIMScheduler,
):
    device = torch.device("cuda")
    dtype = torch.float16

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype).to(
        device
    )

    pipe = pipeline_class.from_pretrained(
        sd_path, controlnet=controlnet, torch_dtype=dtype
    ).to(device)

    pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)

    return pipe
