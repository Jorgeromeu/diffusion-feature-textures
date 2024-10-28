import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

import hydra
import torch
from diffusers import ControlNetModel
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from matplotlib import pyplot as plt
from PIL.Image import Image
from pytorch3d.io import load_obj, load_objs_as_meshes

import text3d2video.wandb_util as wbu
import wandb
from text3d2video.attn_processor import MyAttnProcessor, SaveConfig
from text3d2video.camera_placement import turntable_cameras
from text3d2video.disk_multidict import TensorDiskMultiDict
from text3d2video.generative_rendering.configs import RunConfig
from text3d2video.pipelines.controlnet_pipeline import ControlNetPipeline
from text3d2video.rendering import render_depth_map
from text3d2video.util import ordered_sample
from text3d2video.uv_noise import prepare_uv_initialized_latents


def plot_images(
    images: List[Image], do_multiframe_attn: List[int], target_frame_indices: List[int]
):
    n_cameras = len(images)
    fig, axs = plt.subplots(1, n_cameras, figsize=(n_cameras * 5, 5))
    for i, img in enumerate(images):
        axs[i].imshow(img)
        axs[i].set_xticks([])
        axs[i].set_yticks([])

        if i in target_frame_indices and do_multiframe_attn:
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
    num_cameras: int
    do_multiframe_attn: bool
    target_frame_indices: List[int]
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float


@dataclass
class RunCrossFrameAttnExperimentCfg:
    run: RunConfig
    save_config: SaveCfg
    experiment: CrossFrameAttnExperimentCfg


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

    # setup mesh and cameras
    mesh_path = "data/meshes/cat.obj"
    device = torch.device("cuda")
    mesh = load_objs_as_meshes([mesh_path], device=device)

    _, faces, aux = load_obj(mesh_path)
    verts_uvs = aux.verts_uvs
    faces_uvs = faces.textures_idx

    dist = 2
    n_cameras = cfg.experiment.num_cameras
    cameras = turntable_cameras(
        n_cameras,
        dist=dist,
        start_angle=0,
        stop_angle=90,
        device=device,
    )
    meshes = mesh.extend(len(cameras))

    # render depth maps
    depth_maps = render_depth_map(meshes, cameras)

    gen = torch.Generator(device=device)
    gen.manual_seed(0)

    # setup uv-initialized latents
    latents = prepare_uv_initialized_latents(
        meshes, cameras, verts_uvs, faces_uvs, latent_texture_res=70
    )

    output_folder = Path("outs/tensors")
    if output_folder.exists():
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True)
    tensors_folder = output_folder / "tensors"
    tensors_folder.mkdir()

    tensors_multidict = TensorDiskMultiDict(tensors_folder)

    def pre_step(t, i):
        attn_processor.cur_timestep = t
        attn_processor.cur_timestep_idx = i

    pipe.pre_step_callback = pre_step

    n_inference_steps = cfg.experiment.num_inference_steps

    save_cfg = SaveConfig()
    save_cfg.save_steps = ordered_sample(
        range(n_inference_steps), cfg.save_config.n_save_steps
    )
    save_cfg.module_pahts = cfg.save_config.module_paths

    attn_processor = MyAttnProcessor(pipe.unet)
    attn_processor.save_cfg = save_cfg
    attn_processor.target_frame_indices = cfg.experiment.target_frame_indices
    attn_processor.do_st_extended_attention = cfg.experiment.do_multiframe_attn
    attn_processor.saved_tensors = tensors_multidict
    pipe.unet.set_attn_processor(attn_processor)

    gen = torch.Generator(device=device)
    gen.manual_seed(0)

    prompts = [cfg.experiment.prompt] * n_cameras

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
                cfg.experiment.do_multiframe_attn,
                cfg.experiment.target_frame_indices,
            )
        }
    )

    for i, img in enumerate(images):
        img.save(output_folder / f"image_{i}.png")


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run()
