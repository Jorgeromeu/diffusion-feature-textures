from dataclasses import dataclass
from typing import Any

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

import wandb
import wandb_util.wandb_util as wbu
from text3d2video.artifacts.anim_artifact import AnimationArtifact, AnimationConfig
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
    GenerativeRenderingPipeline,
)
from text3d2video.pipelines.pipeline_utils import (
    ModelConfig,
    load_pipeline_from_model_config,
)
from text3d2video.util import ordered_sample


@dataclass
class RunGenerativeRenderingConfig:
    run: wbu.RunConfig
    prompt: str
    animation: AnimationConfig
    generative_rendering: GenerativeRenderingConfig
    noise_initialization: Any
    model: ModelConfig
    seed: int


class RunGenerativeRendering(wbu.WandbRun):
    job_type = "run_generative_rendering"

    def _run(self, cfg: RunGenerativeRenderingConfig):
        # disable gradients
        torch.set_grad_enabled(False)

        # read animation
        animation = AnimationArtifact.from_wandb_artifact_tag(
            cfg.animation.artifact_tag, download=cfg.run.download_artifacts
        )
        frame_indices = animation.frame_indices(cfg.animation.n_frames)
        cam_frames, mesh_frames = animation.load_frames(frame_indices)
        uv_verts, uv_faces = animation.uv_data()

        # load pipeline
        device = torch.device("cuda")
        pipe = load_pipeline_from_model_config(
            GenerativeRenderingPipeline, cfg.model, device
        )

        noise_initializer = instantiate(cfg.noise_initialization)

        # set seed
        generator = torch.Generator(device=device)
        generator.manual_seed(cfg.seed)

        # inference
        video_frames = pipe(
            cfg.prompt,
            mesh_frames,
            cam_frames,
            uv_verts,
            uv_faces,
            generative_rendering_config=cfg.generative_rendering,
            noise_initializer=noise_initializer,
            generator=generator,
        )

        # save video
        video_artifact = VideoArtifact.create_empty_artifact("video")
        video_artifact.write_frames(video_frames, fps=10)

        # log sampled frames
        sampled_frames = ordered_sample(video_frames, 5)
        wandb.log({"frames": [wandb.Image(frame) for frame in sampled_frames]})

        # log video to run
        wandb.log({"video": wandb.Video(str(video_artifact.get_mp4_path()))})

        # save video artifact
        video_artifact.log_if_enabled()


cs = ConfigStore.instance()
cs.store(name=RunGenerativeRendering.job_type, node=RunGenerativeRenderingConfig)


@hydra.main(config_path="../config", config_name=RunGenerativeRendering.job_type)
def main(cfg: RunGenerativeRenderingConfig):
    run = RunGenerativeRendering()
    run.execute(cfg)


if __name__ == "__main__":
    main()
