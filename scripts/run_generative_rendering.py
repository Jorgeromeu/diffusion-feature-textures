import hydra
import torch
from diffusers import ControlNetModel
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate

import text3d2video.wandb_util as wbu
import wandb
from text3d2video.artifacts.animation_artifact import AnimationArtifact
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.generative_rendering.configs import RunGenerativeRenderingConfig
from text3d2video.generative_rendering.generative_rendering_pipeline import (
    GenerativeRenderingPipeline,
)

cs = ConfigStore.instance()
cs.store(name="generative_rendering", node=RunGenerativeRenderingConfig)


@hydra.main(config_path="../config", config_name="generative_rendering")
def run(cfg: RunGenerativeRenderingConfig):
    do_run = wbu.setup_run(cfg)
    if not do_run:
        return

    torch.set_grad_enabled(False)

    # read animation
    animation = AnimationArtifact.from_wandb_artifact_tag(
        cfg.animation.artifact_tag, download=cfg.run.download_artifacts
    )
    frame_nums = animation.frame_nums(cfg.animation.n_frames)
    mesh_frames = animation.load_frames(frame_nums)
    cameras = animation.cameras(frame_nums)
    uv_verts, uv_faces = animation.texture_data()

    # load pipeline
    device = torch.device("cuda")
    dtype = torch.float16

    sd_repo = cfg.model.sd_repo
    controlnet_repo = cfg.model.controlnet_repo

    controlnet = ControlNetModel.from_pretrained(controlnet_repo, torch_dtype=dtype).to(
        device
    )

    pipe: GenerativeRenderingPipeline = GenerativeRenderingPipeline.from_pretrained(
        sd_repo, controlnet=controlnet, torch_dtype=dtype
    ).to(device)

    pipe.scheduler = instantiate(cfg.model.scheduler).__class__.from_config(
        pipe.scheduler.config
    )

    pipe.module_paths = cfg.generative_rendering.module_paths

    video_frames = pipe(
        cfg.prompt,
        mesh_frames,
        cameras,
        uv_verts,
        uv_faces,
        generative_rendering_config=cfg.generative_rendering,
        noise_initialization_config=cfg.noise_initialization,
        rerun_config=cfg.rerun,
        save_config=cfg.save_tensors,
    )

    if cfg.save_tensors.enabled:
        pipe.log_tensors_artifact()

    # save video
    video_artifact = VideoArtifact.create_empty_artifact(cfg.out_artifact)
    video_artifact.write_frames(video_frames, fps=10)

    # log video to run
    wandb.log({"video": wandb.Video(str(video_artifact.get_mp4_path()))})

    # save video artifact
    video_artifact.log_if_enabled()
    wandb.finish()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run()
