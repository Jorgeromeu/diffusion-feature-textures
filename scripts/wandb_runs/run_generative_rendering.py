from dataclasses import dataclass

import torch

import wandb_util.wandb_util as wbu
from text3d2video.artifacts.anim_artifact import AnimationArtifact
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
from text3d2video.utilities.video_util import pil_frames_to_clip


@dataclass
class RunGenerativeRenderingConfig:
    prompt: str
    animation_tag: str
    generative_rendering: GenerativeRenderingConfig
    model: ModelConfig
    seed: int = 0
    kf_seed: int = 0
    out_artifact: str = "video"


class RunGenerativeRendering(wbu.WandbRun):
    job_type = "run_generative_rendering"

    def _run(self, cfg: RunGenerativeRenderingConfig):
        # disable gradients
        torch.set_grad_enabled(False)

        # read animation
        animation = AnimationArtifact.from_wandb_artifact_tag(
            cfg.animation_tag, download=True
        )
        cam_frames, mesh_frames = animation.load_frames()
        verts_uvs, faces_uvs = animation.uv_data()

        # load pipeline
        device = torch.device("cuda")
        pipe = load_pipeline(
            GenerativeRenderingPipeline, cfg.model.sd_repo, cfg.model.controlnet_repo
        )

        noise_initializer = UVNoiseInitializer(noise_texture_res=120)

        # set seed
        generator = torch.Generator(device=device)
        generator.manual_seed(cfg.seed)

        # set kf seed
        kf_generator = torch.Generator(device=device)
        kf_generator.manual_seed(cfg.kf_seed)

        # inference
        video_frames = pipe(
            cfg.prompt,
            mesh_frames,
            cam_frames,
            verts_uvs,
            faces_uvs,
            conf=cfg.generative_rendering,
            noise_initializer=noise_initializer,
            generator=generator,
            kf_generator=kf_generator,
        )

        # save video
        video_artifact = VideoArtifact.create_empty_artifact("video")
        video_artifact.write_frames(video_frames)

        # log video to run
        wbu.log_moviepy_clip("video", pil_frames_to_clip(video_frames), fps=10)

        # save video artifact
        video_artifact.log_if_enabled()
