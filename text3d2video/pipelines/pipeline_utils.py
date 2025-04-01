from typing import Any

import torch
from attr import dataclass
from diffusers import ControlNetModel
from diffusers.schedulers import DDIMScheduler
from hydra.utils import instantiate


@dataclass
class ModelConfig:
    sd_repo: str
    controlnet_repo: str
    scheduler: Any


def load_pipeline_from_model_config(
    pipeline_class, model_config: ModelConfig, device="cuda"
):
    scheduler_class = instantiate(model_config.scheduler).__class__
    return load_pipeline(
        pipeline_class,
        model_config.sd_repo,
        model_config.controlnet_repo,
        scheduler_class=scheduler_class,
        device=device,
    )


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
