from dataclasses import dataclass

import torch
import torchvision.transforms.functional as TF
from torch import Tensor

import wandb_util.wandb_util as wbu
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.artifacts.texture_artifact import TextureArtifact
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.noise_initialization import UVNoiseInitializer
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
    GenerativeRenderingPipeline,
)
from text3d2video.pipelines.pipeline_utils import (
    ModelConfig,
    load_pipeline,
)
from text3d2video.rendering import render_texture
from text3d2video.utilities.video_util import pil_frames_to_clip


@dataclass
class RenderNoiseGrConfig:
    prompt: str
    animation_tag: str
    texture_tag: str
    generative_rendering: GenerativeRenderingConfig
    model: ModelConfig
    start_noise_level: float = 0.0
    seed: int = 0
    kf_seed: int = 0
    out_artifact: str = "video"


@wbu.wandb_run("render_noise_gr")
def render_noise_gr(cfg: RenderNoiseGrConfig, run_config: wbu.RunConfig):
    # disable gradients
    torch.set_grad_enabled(False)

    # read animation
    anim = AnimationArtifact.from_wandb_artifact_tag(cfg.animation_tag, download=True)
    cams, meshes = anim.load_frames()
    verts_uvs, faces_uvs = anim.uv_data()

    # read texture
    texture = TextureArtifact.from_wandb_artifact_tag(cfg.texture_tag, download=True)
    texture = texture.read_texture()

    # render textured frames
    renders = render_texture(meshes, cams, texture, verts_uvs, faces_uvs)
    renders = [TF.to_pil_image(r) for r in renders]

    # load pipeline
    device = torch.device("cuda")
    pipe = load_pipeline(
        GenerativeRenderingPipeline, cfg.model.sd_repo, cfg.model.controlnet_repo
    )

    # get start t, encode renders and appropriate noise
    start_t = pipe.get_partial_timesteps(
        cfg.generative_rendering.num_inference_steps, cfg.start_noise_level
    )[0]

    if cfg.start_noise_level == 0:
        start_t = Tensor([999]).to(start_t)

    start_latents = pipe.encode_images(renders)
    noise = torch.randn_like(start_latents)
    start_latents = pipe.scheduler.add_noise(start_latents, noise, start_t)

    # set seed
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.seed)

    # set kf seed
    kf_generator = torch.Generator(device=device)
    kf_generator.manual_seed(cfg.kf_seed)

    noise_initializer = UVNoiseInitializer(noise_texture_res=120)

    # inference
    video_frames = pipe(
        cfg.prompt,
        meshes,
        cams,
        verts_uvs,
        faces_uvs,
        noise_initializer=noise_initializer,
        conf=cfg.generative_rendering,
        generator=generator,
        kf_generator=kf_generator,
        start_latents=start_latents,
        start_noise_level=cfg.start_noise_level,
    )

    # save video
    video_artifact = VideoArtifact.create_empty_artifact("video")
    video_artifact.write_frames(video_frames)

    # log video to run
    wbu.log_moviepy_clip("video", pil_frames_to_clip(video_frames), fps=10)

    # save video artifact
    video_artifact.log()
