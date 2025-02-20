import torch
from diffusers import ControlNetModel, DPMSolverMultistepScheduler


def load_pipeline(
    pipeline_class: type,
    sd_path=None,
    controlnet_path=None,
    device="cuda",
    dtype=torch.float16,
    scheduler_class=DPMSolverMultistepScheduler,
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
