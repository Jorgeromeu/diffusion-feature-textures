from dataclasses import dataclass

import torch

import wandb_util.wandb_util as wbu
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.artifacts.extr_frames_artifact import ExtrFramesArtifact
from text3d2video.mip import seq_max_uv_res
from text3d2video.pipelines.pipeline_utils import (
    ModelConfig,
    load_pipeline,
)
from text3d2video.pipelines.texturing_pipeline import TexGenConfig, TexGenPipeline
from text3d2video.utilities.video_util import pil_frames_to_clip


@dataclass
class TexGenExtrConfig:
    prompt: str
    animation_tag: str
    model: ModelConfig
    texgen: TexGenConfig
    seed: int = 0
    extr_out_art: str = "extr_frames"


@wbu.wandb_run("run_texgen_extr")
def run_texgen_extr(cfg: TexGenExtrConfig, run_config: wbu.RunConfig):
    # disable gradients
    torch.set_grad_enabled(False)

    # read animation
    anim = AnimationArtifact.from_wandb_artifact_tag(cfg.animation_tag, download=True)
    seq = anim.read_anim_seq()

    # load pipeline
    device = torch.device("cuda")
    pipe: TexGenPipeline = load_pipeline(
        TexGenPipeline, cfg.model.sd_repo, cfg.model.controlnet_repo
    )

    # set seed
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    # determine uv res
    uv_res = seq_max_uv_res(seq, resolution=512)

    out = pipe(
        cfg.prompt,
        seq,
        conf=cfg.texgen,
        uv_res=uv_res,
        generator=generator,
    )

    # log video to run
    wbu.log_moviepy_clip("video", pil_frames_to_clip(out.images), fps=10)

    # log extraction artifact
    extr_art = ExtrFramesArtifact.create_empty_artifact(cfg.extr_out_art)
    extr_art.write(out.images, out.latents, out.texture)
    extr_art.log()
