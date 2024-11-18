from dataclasses import dataclass
from typing import List

import hydra
import matplotlib
import torch
from diffusers import ControlNetModel
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from PIL.Image import Image

import text3d2video.wandb_util as wbu
import wandb
from text3d2video.artifacts.animation_artifact import AnimationArtifact
from text3d2video.artifacts.attn_features_artifact import AttentionFeaturesArtifact
from text3d2video.attn_processor import MultiFrameAttnProcessor, SaveConfig
from text3d2video.camera_placement import turntable_cameras
from text3d2video.generative_rendering.configs import (
    AnimationConfig,
    NoiseInitializationConfig,
    RunConfig,
)
from text3d2video.pipelines.controlnet_pipeline import ControlNetPipeline
from text3d2video.rendering import render_depth_map
from text3d2video.util import ordered_sample
from text3d2video.uv_noise import prepare_latents

matplotlib.use("Agg")


def plot_images(images: List[Image], target_frame_indices: List[int]):
    n_cameras = len(images)
    fig, axs = plt.subplots(1, n_cameras, figsize=(n_cameras * 5, 5))
    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

        if i in target_frame_indices:
            for spine in axs[i].spines.values():
                spine.set_edgecolor("red")
                spine.set_linewidth(5)
        else:
            axs[i].axis("off")
    plt.tight_layout()
    return fig


@dataclass
class SaveCfg:
    enabled: bool
    module_paths: List[str]
    n_save_steps: int
    out_artifact: str


@dataclass
class CrossFrameAttnExperimentCfg:
    seed: int
    prompt: str
    n_views: int
    attend_to_self: bool
    target_frame_indices: List[int]
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float


@dataclass
class RunCrossFrameAttnExperimentCfg:
    run: RunConfig
    save_config: SaveCfg
    animation: AnimationConfig
    noise_initialization: NoiseInitializationConfig
    cross_frame_attn_experiment: CrossFrameAttnExperimentCfg


cs = ConfigStore.instance()
cs.store(name="cross_frame_attn_experiment", node=RunCrossFrameAttnExperimentCfg)


@hydra.main(config_path="../config", config_name="cross_frame_attn_experiment")
def run(cfg: RunCrossFrameAttnExperimentCfg):
    do_run = wbu.setup_run(cfg)
    if not do_run:
        return

    torch.set_grad_enabled(False)

    # load pipeline
    device = torch.device("cuda")
    dtype = torch.float16

    sd_repo = cfg.model.sd_repo
    controlnet_repo = cfg.model.controlnet_repo

    # load pipeline
    controlnet = ControlNetModel.from_pretrained(controlnet_repo, torch_dtype=dtype).to(
        device
    )

    pipe = ControlNetPipeline.from_pretrained(
        sd_repo, controlnet=controlnet, torch_dtype=dtype
    ).to(device)

    pipe.scheduler = instantiate(cfg.model.scheduler).__class__.from_config(
        pipe.scheduler.config
    )

    # read animation

    # read animation
    animation = AnimationArtifact.from_wandb_artifact_tag(
        cfg.animation.artifact_tag, download=cfg.run.download_artifacts
    )
    frame_nums = animation.frame_nums(cfg.animation.n_frames)
    meshes = animation.load_frames(frame_nums)
    cameras = animation.cameras(frame_nums)
    verts_uvs, faces_uvs = animation.texture_data()

    dist = 2
    n_cameras = cfg.cross_frame_attn_experiment.n_views
    cameras = turntable_cameras(
        n_cameras,
        dist=dist,
        start_angle=0,
        stop_angle=90,
        device=device,
    )

    # render depth maps
    depth_maps = render_depth_map(meshes, cameras)

    gen = torch.Generator(device=device)
    gen.manual_seed(cfg.cross_frame_attn_experiment.seed)

    # setup uv-initialized latents
    latents = prepare_latents(
        meshes,
        cameras,
        verts_uvs,
        faces_uvs,
        cfg.noise_initialization,
        generator=gen,
    )

    out_artifact: AttentionFeaturesArtifact = (
        AttentionFeaturesArtifact.create_empty_artifact(cfg.save_config.out_artifact)
    )

    tensors_multidict = out_artifact.create_features_diskdict()

    attn_processor = MultiFrameAttnProcessor(pipe.unet)

    def pre_step(t, i):
        attn_processor.cur_timestep = t
        attn_processor.cur_timestep_idx = i

    pipe.pre_step_callback = pre_step

    n_inference_steps = cfg.cross_frame_attn_experiment.num_inference_steps

    save_cfg = SaveConfig()
    save_cfg.save_steps = ordered_sample(
        range(n_inference_steps), cfg.save_config.n_save_steps
    )
    save_cfg.module_pahts = cfg.save_config.module_paths

    attn_processor.save_cfg = save_cfg
    attn_processor.target_frame_indices = (
        cfg.cross_frame_attn_experiment.target_frame_indices
    )
    attn_processor.attend_to_self = cfg.cross_frame_attn_experiment.attend_to_self
    attn_processor.saved_tensors = tensors_multidict
    pipe.unet.set_attn_processor(attn_processor)

    gen = torch.Generator(device=device)
    gen.manual_seed(0)

    prompts = [cfg.cross_frame_attn_experiment.prompt] * n_cameras

    images = pipe(
        prompts,
        depth_maps,
        generator=gen,
        initial_latents=latents,
        num_inference_steps=n_inference_steps,
    )

    # plot images
    wandb.log(
        {
            "images": plot_images(
                images,
                cfg.cross_frame_attn_experiment.target_frame_indices,
            )
        }
    )

    out_artifact.write_images(images)
    out_artifact.log_if_enabled()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run()
