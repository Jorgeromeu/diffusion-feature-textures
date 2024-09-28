from typing import Tuple

import hydra
import rerun as rr
import torchvision.transforms.functional as TF
from diffusers import StableDiffusionControlNetPipeline
from omegaconf import DictConfig
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Meshes

import text3d2video.rerun_util as ru
import text3d2video.wandb_util as wu
import wandb
from text3d2video.artifacts.animation_artifact import AnimationArtifact
from text3d2video.artifacts.multiview_features_artifact import MVFeaturesArtifact
from text3d2video.diffusion import depth2img, depth2img_pipe
from text3d2video.multidict import MultiDict
from text3d2video.rendering import make_rasterizer, normalize_depth_map
from text3d2video.sd_feature_extraction import DiffusionFeatureExtractor
from text3d2video.util import multiview_cameras


def extract_multiview_features(
    pipe: StableDiffusionControlNetPipeline,
    mesh: Meshes,
    prompt: str = "Deadpool",
    n_views=9,
    resolution=512,
    device="cpu",
    num_inference_steps=30,
    save_steps=None,
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

    # setup feature extractor
    extractor = DiffusionFeatureExtractor()
    extractor.add_save_feature_hook("level_0", pipe.unet.up_blocks[0])
    extractor.add_save_feature_hook("level_1", pipe.unet.up_blocks[1])
    extractor.add_save_feature_hook("level_2", pipe.unet.up_blocks[2])
    extractor.add_save_feature_hook("level_3", pipe.unet.up_blocks[3])
    if save_steps is None:
        save_steps = []
    extractor.save_steps = save_steps

    # Generate images
    prompts = [prompt] * n_views
    generted_ims = depth2img(
        pipe, prompts, depth_imgs, num_inference_steps=num_inference_steps
    )

    # log generated images
    rr_seq.step()
    for view_i in range(n_views):
        rr.log(f"cam_{view_i}", rr.Image(generted_ims[view_i]))

    # collect features
    features = MultiDict()

    for name in extractor.hook_manager.named_hooks():
        for timestep in extractor.save_steps:

            # get all features for name and timestep
            extracted_features = extractor.get_feature(name, timestep=timestep)

            for view_i in range(n_views):

                key = {"view": view_i, "timestep": timestep, "layer": name}
                features.add(key, extracted_features[view_i])

    return cameras, features, generted_ims


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

    # load pipeline
    pipe = depth2img_pipe(device=device)

    # get multiview diffusion features
    cams, features, ims = extract_multiview_features(
        pipe,
        mesh,
        prompt=mv_cfg.prompt,
        n_views=mv_cfg.num_views,
        num_inference_steps=mv_cfg.num_inference_steps,
        device=device,
        save_steps=mv_cfg.save_steps,
    )

    # save features as artifact
    out_artifact = MVFeaturesArtifact.create_wandb_artifact(
        mv_cfg.out_artifact_name, cameras=cams, features=features, images=ims
    )

    wu.log_artifact_if_enabled(out_artifact)

    wandb.finish()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run()
