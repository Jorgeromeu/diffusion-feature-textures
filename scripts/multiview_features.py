from pathlib import Path
import shutil
import time
from einops import repeat
import torch
from diffusers import StableDiffusionControlNetPipeline
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import look_at_view_transform, RasterizationSettings, MeshRasterizer
import rerun as rr
import rerun.blueprint as rrb
import wandb
from text3d2video.diffusion import depth2img, depth2img_pipe
from text3d2video.rendering import make_rasterizer, normalize_depth_map
import text3d2video.rerun_util as ru
import torchvision.transforms.functional as TF
import faiss
from PIL import Image
from text3d2video.sd_feature_extraction import SDFeatureExtractor
from text3d2video.util import project_vertices_to_features, multiview_cameras, random_solid_color_img
from text3d2video.visualization import RgbPcaUtil
from text3d2video.wandb_util import MVFeaturesArtifact


def extract_multiview_features(
        pipe: StableDiffusionControlNetPipeline,
        mesh: Meshes,
        prompt: str = 'Deadpool',
        n_views=9,
        resolution=512,
        device='cpu',
) -> torch.Tensor:
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
    for i in range(n_views):
        ru.log_pt3d_FovCamrea(f'cam_{i}', cameras, batch_idx=i, res=resolution)

    # render depth maps
    rasterizer = make_rasterizer(cameras, resolution)

    batch_mesh = mesh.extend(n_views)
    fragments = rasterizer(batch_mesh)

    depth_maps = normalize_depth_map(fragments.zbuf)
    depth_imgs = [TF.to_pil_image(depth_maps[i, :, :, 0])
                  for i in range(n_views)]

    rr_seq.step()

    # log depth images
    for i in range(n_views):
        rr.log(f'cam_{i}', rr.Image(depth_imgs[i]))

    # setup feature extractor
    feature_extractor = SDFeatureExtractor(pipe)

    # Generate images
    prompts = [prompt] * n_views
    generted_ims = depth2img(pipe, prompts, depth_imgs, num_inference_steps=30)

    rr_seq.step()

    # log generated images
    for i in range(n_views):
        rr.log(f'cam_{i}', rr.Image(generted_ims[i]))

    # extract features
    # size = n_views
    extracted_features = feature_extractor.get_feature(level=0, timestep=20)

    feature_maps = []
    for i in range(n_views):
        feature_map = torch.Tensor(extracted_features[i])
        feature_maps.append(feature_map)

    return cameras, feature_maps, generted_ims


if __name__ == "__main__":

    animation_art = 'backflip:latest'
    prompt = 'Deadpool'
    n_views = 10
    out_artifact_name = 'multiview_features'

    wandb.init(project='diffusion-3d-features', job_type='multiview_features')

    # download animation
    animation_artifact = wandb.use_artifact(animation_art)
    animation_dir = Path(animation_artifact.download())
    mesh_path = animation_dir / 'static.obj'

    # init 3D logging
    rr.init('multiview_features')
    rr.serve()
    ru.pt3d_setup()

    # log blueprint
    blueprint = rrb.Blueprint(rrb.Spatial3DView())
    rr.send_blueprint(blueprint)

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

    out_artifact = MVFeaturesArtifact.create(
        out_artifact_name,
        cams,
        features,
        ims
    )

    wandb.log_artifact(out_artifact)
