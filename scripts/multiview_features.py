from pathlib import Path
from typing import List, Tuple
from codetiming import Timer

import torch
import hydra
import rerun as rr
import torchvision.transforms.functional as TF
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from omegaconf import DictConfig
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Meshes

from text3d2video.disk_multidict import TensorDiskMultiDict
import text3d2video.rerun_util as ru
import text3d2video.wandb_util as wu
import wandb
from text3d2video.artifacts.animation_artifact import AnimationArtifact
from text3d2video.artifacts.multiview_features_artifact import MVFeaturesArtifact
from text3d2video.pipelines.my_pipeline import MyPipeline
from text3d2video.multidict import MultiDict
from text3d2video.rendering import make_rasterizer, normalize_depth_map
from text3d2video.sd_feature_extraction import (
    DiffusionFeatureExtractor,
    get_module_from_path,
)
from text3d2video.util import multiview_cameras


def extract_multiview_features(
    features_multidict: TensorDiskMultiDict,
    pipe: StableDiffusionControlNetPipeline,
    mesh: Meshes,
    prompt: str = "Deadpool",
    n_views=9,
    resolution=512,
    device="cuda",
    num_inference_steps=30,
    save_steps: List[int] = None,
    module_paths: List[str] = None,
) -> Tuple[FoVPerspectiveCameras, MultiDict, list]:
    """
    Compute Diffusion 3D Features for a given mesh, and represent them as vertex features.
    :param pipe: Depth 2 Image Diffusion pipeline to extract features from
    :param mesh: Pytorch3D Meshes object representing the mesh
    :param prompt: Prompt to generate images from
    :param n_views: Number of views to render depth maps from
    :param resolution: Resolution of depth maps, and generated images
    :param device: Device to run the computation
    :return: Vertex features representing the diffusion features
    """

    if save_steps is None:
        save_steps = []

    if module_paths is None:
        module_paths = []

    # manual timesteps
    rr_seq = ru.TimeSequence("steps")

    # log original mesh
    rr.log("mesh", ru.pt3d_mesh(mesh))

    # generate cameras
    cameras = multiview_cameras(mesh, n_views, device=device)
    n_views = len(cameras)

    # log cameras
    rr_seq.step()
    for view_i in range(n_views):
        ru.log_pt3d_FovCamrea(
            f"cam_{view_i}", cameras, batch_idx=view_i, res=resolution
        )

    # render depth maps
    rasterizer = make_rasterizer(cameras, resolution)
    batch_mesh = mesh.extend(n_views)
    fragments = rasterizer(batch_mesh)
    depth_maps = normalize_depth_map(fragments.zbuf)
    depth_imgs = [TF.to_pil_image(depth_maps[i, :, :, 0]) for i in range(n_views)]

    # log depth maps
    rr_seq.step()
    for view_i in range(n_views):
        rr.log(f"cam_{view_i}", rr.Image(depth_imgs[view_i]))

    # add hooks
    extractor = DiffusionFeatureExtractor()
    for module_path in module_paths:
        module = get_module_from_path(pipe.unet, module_path)
        extractor.add_save_feature_hook(module_path, module)

    extractor.save_steps = save_steps

    # Generate images
    prompts = [prompt] * n_views
    with Timer(initial_text="Generating images"):
        generted_ims = pipe(
            prompts, depth_imgs, num_inference_steps=num_inference_steps
        )

    # log generated images
    rr_seq.step()
    for view_i in range(n_views):
        rr.log(f"cam_{view_i}", rr.Image(generted_ims[view_i]))

    with Timer(initial_text="Saving Features to disk"):
        for name in extractor.hook_manager.named_hooks():
            for timestep in extractor.save_steps:

                # get all features for name and timestep
                extracted_features = extractor.get_feature(name, timestep=timestep)

                for view_i in range(n_views):

                    key = {"view": view_i, "timestep": timestep, "layer": name}
                    features_multidict[key] = torch.Tensor(extracted_features[view_i])

    return cameras, generted_ims


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run(cfg: DictConfig):

    # read config
    mv_cfg = cfg.multiview_feature_extraction

    # init wandb
    wu.init_run(dev_run=cfg.dev_run, job_type="multiview_features")
    wandb.config.update(dict(mv_cfg))

    # init 3D logging
    ru.set_logging_state(cfg.rerun_enabled)
    rr.init("multiview_features")
    rr.serve()
    ru.pt3d_setup()

    # load animation artifact
    animation_artifact = wu.get_artifact(mv_cfg.animation_artifact_tag)
    animation_artifact = AnimationArtifact.from_wandb_artifact(animation_artifact)

    # get mesh
    device = "cuda:0"
    mesh = animation_artifact.load_static_mesh(device)

    # load depth2img pipeline
    dtype = torch.float16
    sd_repo = cfg.stable_diffusion.name
    controlnet_repo = cfg.controlnet.name
    device = torch.device("cuda")

    controlnet = ControlNetModel.from_pretrained(
        controlnet_repo, torch_dtype=torch.float16
    ).to(device)

    pipe = MyPipeline.from_pretrained(
        sd_repo, controlnet=controlnet, torch_dtype=dtype
    ).to(device)

    # get multiview diffusion features
    features_multidict = TensorDiskMultiDict(Path("outs/multiview_features"))
    features_multidict.clear()
    cams, ims = extract_multiview_features(
        features_multidict,
        pipe,
        mesh,
        prompt=mv_cfg.prompt,
        n_views=mv_cfg.num_views,
        num_inference_steps=mv_cfg.num_inference_steps,
        device=device,
        save_steps=mv_cfg.save_steps,
        module_paths=mv_cfg.module_paths,
    )

    # # save features as artifact
    # out_artifact = MVFeaturesArtifact.create_wandb_artifact(
    #     mv_cfg.out_artifact_name,
    #     cameras=cams,
    #     features_path=features_multidict.path,
    #     images=ims,
    # )

    # wu.log_artifact_if_enabled(out_artifact)
    # wandb.finish()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run()
