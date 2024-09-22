from pathlib import Path
from typing import Tuple

import rerun as rr
import torchvision.transforms.functional as TF
from diffusers import StableDiffusionControlNetPipeline
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Meshes

import text3d2video.rerun_util as ru
import wandb
from text3d2video.artifacts.multiview_features_artifact import MVFeaturesArtifact
from text3d2video.diffusion import depth2img, depth2img_pipe
from text3d2video.multidict import MultiDict
from text3d2video.rendering import make_rasterizer, normalize_depth_map
from text3d2video.sd_feature_extraction import DiffusionFeatureExtractor
from text3d2video.util import multiview_cameras


def extract_multiview_features(
    pipe: StableDiffusionControlNetPipeline,
    mesh: Meshes,
    prompt: str = 'Deadpool',
    n_views=9,
    resolution=512,
    device='cpu',
    num_inference_steps=30
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
    rr.log('mesh', ru.pt3d_mesh(mesh))

    rr_seq.step()
    # generate cameras
    cameras = multiview_cameras(mesh, n_views, device=device)
    n_views = len(cameras)

    # log cameras
    for view_i in range(n_views):
        ru.log_pt3d_FovCamrea(
            f'cam_{view_i}', cameras, batch_idx=view_i, res=resolution)

    # render depth maps
    rasterizer = make_rasterizer(cameras, resolution)

    batch_mesh = mesh.extend(n_views)
    fragments = rasterizer(batch_mesh)

    depth_maps = normalize_depth_map(fragments.zbuf)
    depth_imgs = [TF.to_pil_image(depth_maps[i, :, :, 0])
                  for i in range(n_views)]

    rr_seq.step()

    # log depth images
    for view_i in range(n_views):
        rr.log(f'cam_{view_i}', rr.Image(depth_imgs[view_i]))

    # setup feature extractor
    extractor = DiffusionFeatureExtractor()
    extractor.add_save_feature_hook('level_0', pipe.unet.up_blocks[0])
    extractor.add_save_feature_hook('level_1', pipe.unet.up_blocks[1])
    extractor.add_save_feature_hook('level_2', pipe.unet.up_blocks[2])
    extractor.add_save_feature_hook('level_3', pipe.unet.up_blocks[3])

    # Generate images
    prompts = [prompt] * n_views
    generted_ims = depth2img(pipe, prompts, depth_imgs, num_inference_steps=num_inference_steps)

    rr_seq.step()

    # log generated images
    for view_i in range(n_views):
        rr.log(f'cam_{view_i}', rr.Image(generted_ims[view_i]))

    features = MultiDict()

    for name in extractor.hook_manager.named_hooks():
        for timestep in extractor.save_steps:

            # get all features for name and timestep
            extracted_features = extractor.get_feature(name, timestep=timestep)

            for view_i in range(n_views):

                key = {'view': view_i, 'timestep': timestep, 'layer': name}
                features.add(key, extracted_features[view_i])

    return cameras, features, generted_ims

if __name__ == "__main__":

    animation_art = 'backflip:latest'
    prompt = 'Deadpool'
    n_views = 10
    out_artifact_name = 'deadpool_mv_features'
    n_inf_steps = 30

    wandb.init(project='diffusion-3d-features', job_type='multiview_features')

    # download animation
    animation_artifact = wandb.use_artifact(animation_art)
    animation_dir = Path(animation_artifact.download())
    mesh_path = animation_dir / 'static.obj'

    # init 3D logging
    rr.init('multiview_features')
    rr.serve()
    ru.pt3d_setup()

    # load mesh
    device = 'cuda:0'
    mesh: Meshes = load_objs_as_meshes([mesh_path], device=device)

    # load pipeline
    pipe = depth2img_pipe(device=device)

    # compute diffusion features
    cams, features, ims = extract_multiview_features(
        pipe,
        mesh,
        device=device,
        n_views=n_views,
        prompt=prompt
    )

    out_artifact = MVFeaturesArtifact.create_wandb_artifact(
        out_artifact_name,
        cameras=cams,
        features=features,
        images=ims
    )

    wandb.log_artifact(out_artifact)
