from dataclasses import dataclass
from typing import Any

import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import FoVPerspectiveCameras

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
from text3d2video.utilities.camera_placement import turntable_extrinsics


@dataclass
class RunReposableDiffusionT2VConfig:
    run: wbu.RunConfig
    seed: int
    prompt: str
    animation: AnimationConfig
    reposable_diffusion: ReposableDiffusionConfig
    num_views: int
    noise_initialization: Any
    model: ModelConfig


class RunReposableDiffusionT2V(wbu.WandbRun):
    job_type = "run_reposable_diffusion_t2v"

    def _run(self, cfg: RunReposableDiffusionT2VConfig):
        # disable gradients
        torch.set_grad_enabled(False)

        # load pipeline
        device = torch.device("cuda")
        pipe = load_pipeline_from_model_config(
            ReposableDiffusionPipeline, cfg.model, device
        )

        # read input animation
        target_animation = AnimationArtifact.from_wandb_artifact_tag(
            cfg.animation.artifact_tag, download=cfg.run.download_artifacts
        )
        uv_verts, uv_faces = target_animation.uv_data()
        frame_indices = target_animation.frame_indices(cfg.animation.n_frames)
        frame_cams, frame_meshes = target_animation.load_frames(frame_indices)

        # source frames: Multiview
        angles = np.linspace(0, 360, cfg.num_views, endpoint=False)
        R, T = turntable_extrinsics(angles=angles, dists=1.5)
        aggr_cams = FoVPerspectiveCameras(R=R, T=T, device="cuda", fov=65)
        mesh_path = "data/meshes/mixamo-human.obj"
        mesh = load_objs_as_meshes([mesh_path], device=device)
        aggr_meshes = mesh.extend(len(aggr_cams))

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


cs = ConfigStore.instance()
cs.store(name=RunReposableDiffusionT2V.job_type, node=RunReposableDiffusionT2VConfig)


@hydra.main(config_path="../config", config_name=RunReposableDiffusionT2V.job_type)
def main(cfg: RunReposableDiffusionT2VConfig):
    run = RunReposableDiffusionT2V()
    run.execute(cfg)


if __name__ == "__main__":
    main()
