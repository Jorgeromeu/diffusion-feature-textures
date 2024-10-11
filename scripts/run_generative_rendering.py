import hydra
import torch
from diffusers import ControlNetModel
from omegaconf import DictConfig

import text3d2video.wandb_util as wbu
import wandb
from text3d2video.artifacts.animation_artifact import AnimationArtifact
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.generative_rendering.generative_rendering import (
    GenerativeRenderingPipeline,
)


@hydra.main(config_path="../config", config_name="generative_rendering")
def run(cfg: DictConfig):

    wbu.init_run(dev_run=True, job_type="generate_video", tags=[])

    device = torch.device("cuda")
    dtype = torch.float16

    sd_repo = cfg.model.sd_repo
    controlnet_repo = cfg.model.controlnet_repo

    controlnet = ControlNetModel.from_pretrained(controlnet_repo, torch_dtype=dtype).to(
        device
    )

    pipe = GenerativeRenderingPipeline.from_pretrained(
        sd_repo, controlnet=controlnet, torch_dtype=dtype
    ).to(device)

    # read animation
    animation = AnimationArtifact.from_wandb_artifact_tag(
        cfg.inputs.animation_artifact_tag, download=False
    )
    frame_nums = animation.frame_nums(cfg.inputs.animation_n_frames)

    # read frames
    mesh_frames = animation.load_frames(frame_nums)
    cameras = animation.cameras(frame_nums)

    generator = torch.Generator(device=device)
    generator.manual_seed(cfg.inputs.sd_seed)

    uv_verts, uv_faces = animation.texture_data()

    # set attention module paths
    pipe.module_paths = cfg.generative_rendering.module_paths

    frames = pipe(
        cfg.inputs.prompt,
        mesh_frames,
        cameras,
        uv_verts,
        uv_faces,
        num_inference_steps=cfg.generative_rendering.num_inference_steps,
        generator=generator,
        num_keyframes=cfg.generative_rendering.num_keyframes,
        chunk_size=cfg.generative_rendering.chunk_size,
        do_pre_attn_injection=cfg.generative_rendering.do_pre_attn_injection,
        do_post_attn_injection=cfg.generative_rendering.do_post_attn_injection,
        rerun=False,
    )

    # save video
    video_artifact = VideoArtifact.create_empty_artifact(cfg.out_artifact)
    video_artifact.write_frames(frames, fps=10)

    # log video to run
    wandb.log({"video": wandb.Video(str(video_artifact.get_mp4_path()))})

    # save video artifact
    video_artifact.log()
    wandb.finish()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run()
