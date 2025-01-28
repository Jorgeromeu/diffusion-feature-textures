import warnings
from dataclasses import dataclass
from typing import Any

import hydra
import torch
from diffusers import ControlNetModel
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

import text3d2video.wandb_util as wbu
import wandb
from scripts.run_generative_rendering import ModelConfig
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.artifacts.gr_data import GrSaveConfig
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.generative_rendering.configs import (
    AnimationConfig,
    GenerativeRenderingConfig,
    RunConfig,
)
from text3d2video.generative_rendering.reposable_diffusion_pipeline import (
    ReposableDiffusionPipeline,
)


@dataclass
class RunReposableDiffusionConfig:
    run: RunConfig
    video_artifact: str
    aggr_artifact: str
    prompt: str
    target_frames: AnimationConfig
    source_frames: AnimationConfig
    save_tensors: GrSaveConfig
    generative_rendering: GenerativeRenderingConfig
    noise_initialization: Any
    model: ModelConfig


JOB_TYPE = "run_reposable_diffusion"

cs = ConfigStore.instance()
cs.store(name=JOB_TYPE, node=RunReposableDiffusionConfig)


@hydra.main(config_path="../config", config_name=JOB_TYPE)
def run(cfg: RunReposableDiffusionConfig):
    # init wandb
    cfg.run.job_type = JOB_TYPE
    do_run = wbu.setup_run(cfg.run, cfg)
    if not do_run:
        return

    # supress warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")

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
    dtype = torch.float16

    sd_repo = cfg.model.sd_repo
    controlnet_repo = cfg.model.controlnet_repo

    controlnet = ControlNetModel.from_pretrained(controlnet_repo, torch_dtype=dtype).to(
        device
    )

    pipe: ReposableDiffusionPipeline = ReposableDiffusionPipeline.from_pretrained(
        sd_repo, controlnet=controlnet, torch_dtype=dtype
    ).to(device)

    pipe.scheduler = instantiate(cfg.model.scheduler).__class__.from_config(
        pipe.scheduler.config
    )

    noise_initializer = instantiate(cfg.noise_initialization)

    # inference
    video_frames = pipe(
        cfg.prompt,
        frame_meshes,
        frame_cams,
        aggr_meshes,
        aggr_cams,
        uv_verts,
        uv_faces,
        generative_rendering_config=cfg.generative_rendering,
        noise_initializer=noise_initializer,
        gr_save_config=cfg.save_tensors,
    )

    vid_frames = video_frames[0 : len(frame_cams)]
    aggr_frames = video_frames[len(frame_cams) :]

    if cfg.save_tensors.enabled:
        pipe.log_data_artifact()

    # save video
    video_artifact = VideoArtifact.create_empty_artifact(cfg.video_artifact)
    video_artifact.write_frames(vid_frames, fps=10)
    video_artifact.log_if_enabled()

    # save aggregation video
    aggr_artifact = VideoArtifact.create_empty_artifact(cfg.aggr_artifact)
    aggr_artifact.write_frames(aggr_frames, fps=10)
    aggr_artifact.log_if_enabled()

    # log videos to run
    wandb.log({"video": wandb.Video(str(video_artifact.get_mp4_path()))})
    wandb.log({"aggr": wandb.Video(str(aggr_artifact.get_mp4_path()))})

    # terminate run
    run = wandb.run
    wandb.finish()
    return run


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run()
