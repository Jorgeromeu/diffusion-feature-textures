from dataclasses import dataclass
from typing import Any

import hydra
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

import wandb
import wandb_util.wandb_util as wbu
from scripts.run_generative_rendering import ModelConfig
from text3d2video.artifacts.anim_artifact import AnimationArtifact, AnimationConfig
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.pipelines.pipeline_utils import load_pipeline_from_model_config
from text3d2video.pipelines.reposable_diffusion_pipeline import (
    ReposableDiffusionConfig,
    ReposableDiffusionPipeline,
)


@dataclass
class RunReposableDiffusionConfig:
    run: wbu.RunConfig
    seed: int
    prompt: str
    target_frames: AnimationConfig
    source_frames: AnimationConfig
    reposable_diffusion: ReposableDiffusionConfig
    noise_initialization: Any
    model: ModelConfig


class RunReposableDiffusion(wbu.WandbRun):
    job_type = "run_reposable_diffusion"

    def _run(self, cfg: RunReposableDiffusionConfig):
        # disable gradients
        torch.set_grad_enabled(False)

        # read animations
        target_animation = AnimationArtifact.from_wandb_artifact_tag(
            cfg.target_frames.artifact_tag, download=cfg.run.download_artifacts
        )
        uv_verts, uv_faces = target_animation.uv_data()
        frame_indices = target_animation.frame_indices(cfg.target_frames.n_frames)
        frame_cams, frame_meshes = target_animation.load_frames(frame_indices)

        source_animation = AnimationArtifact.from_wandb_artifact_tag(
            cfg.source_frames.artifact_tag, download=cfg.run.download_artifacts
        )
        frame_indices = source_animation.frame_indices(cfg.source_frames.n_frames)
        aggr_cams, aggr_meshes = source_animation.load_frames(frame_indices)

        # load pipeline
        device = torch.device("cuda")
        pipe = load_pipeline_from_model_config(
            ReposableDiffusionPipeline, cfg.model, device
        )

        noise_initializer = instantiate(cfg.noise_initialization)

        generator = torch.Generator(device=device)
        generator.manual_seed(cfg.seed)

        # inference
        video_frames = pipe(
            cfg.prompt,
            frame_meshes,
            frame_cams,
            aggr_meshes,
            aggr_cams,
            uv_verts,
            uv_faces,
            reposable_diffusion_config=cfg.reposable_diffusion,
            noise_initializer=noise_initializer,
            generator=generator,
        )

        vid_frames = video_frames[0 : len(frame_cams)]
        aggr_frames = video_frames[len(frame_cams) :]

        # save video
        video_artifact = VideoArtifact.create_empty_artifact("video")
        video_artifact.write_frames(vid_frames, fps=10)
        video_artifact.log_if_enabled()

        # save aggregation video
        aggr_artifact = VideoArtifact.create_empty_artifact("aggr")
        aggr_artifact.write_frames(aggr_frames, fps=10)
        aggr_artifact.log_if_enabled()

        # log videos to run
        wandb.log({"video": wandb.Video(str(video_artifact.get_mp4_path()))})
        wandb.log({"aggr": wandb.Video(str(aggr_artifact.get_mp4_path()))})


job_type = RunReposableDiffusion.job_type
cs = ConfigStore.instance()
cs.store(name=job_type, node=RunReposableDiffusionConfig)


@hydra.main(config_path="../config", config_name=job_type)
def main(cfg: RunReposableDiffusionConfig):
    run = RunReposableDiffusion()
    run.execute(cfg)


if __name__ == "__main__":
    main()
