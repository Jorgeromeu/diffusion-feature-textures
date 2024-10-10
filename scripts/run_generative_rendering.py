import hydra
from omegaconf import DictConfig
import wandb

import torch
from text3d2video.artifacts.animation_artifact import AnimationArtifact
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.generative_rendering import GenerativeRenderingPipeline
from diffusers import ControlNetModel
import text3d2video.wandb_util as wbu


@hydra.main(config_path="../config", config_name="config")
def run(_: DictConfig):

    wbu.init_run(dev_run=False, job_type="generate_video", tags=[])

    sd_repo = "runwayml/stable-diffusion-v1-5"
    controlnet_repo = "lllyasviel/control_v11f1p_sd15_depth"

    device = torch.device("cuda")
    dtype = torch.float16

    controlnet = ControlNetModel.from_pretrained(controlnet_repo, torch_dtype=dtype).to(
        device
    )

    pipe = GenerativeRenderingPipeline.from_pretrained(
        sd_repo, controlnet=controlnet, torch_dtype=dtype
    ).to(device)

    pipe.module_paths = [
        "down_blocks.0.attentions.0.transformer_blocks.0.attn1",
        "down_blocks.0.attentions.1.transformer_blocks.0.attn1",
        "down_blocks.1.attentions.0.transformer_blocks.0.attn1",
        "down_blocks.1.attentions.1.transformer_blocks.0.attn1",
        "down_blocks.2.attentions.0.transformer_blocks.0.attn1",
        "down_blocks.2.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1",
        "mid_block.attentions.0.transformer_blocks.0.attn1",
    ]

    # read animation
    animation_tag = "backflip:latest"
    animation = AnimationArtifact.from_wandb_artifact_tag(animation_tag, download=True)

    # setup frames
    mesh_frames = animation.load_frames(animation.frame_nums(20))

    generator = torch.Generator(device=device)
    generator.manual_seed(0)

    uv_verts, uv_faces = animation.texture_data()

    frames = pipe(
        "Deadpool",
        mesh_frames,
        uv_verts,
        uv_faces,
        num_inference_steps=50,
        generator=generator,
        num_keyframes=6,
        rerun=False,
    )

    # save video
    video_artifact = VideoArtifact.create_empty_artifact("deadpool-joyful-jump")
    video_artifact.write_frames(frames, fps=10)

    # log video to run
    wandb.log({"video": wandb.Video(str(video_artifact.get_mp4_path()))})

    # save video artifact
    video_artifact.log()
    wandb.finish()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run()
