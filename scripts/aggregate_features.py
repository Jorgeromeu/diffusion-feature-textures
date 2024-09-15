from pathlib import Path
import time
from einops import rearrange, repeat
import torch
from diffusers import StableDiffusionControlNetPipeline
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import look_at_view_transform, RasterizationSettings, MeshRasterizer
import rerun as rr
import rerun.blueprint as rrb
import wandb
from text3d2video.diffusion import depth2img, depth2img_pipe
from text3d2video.rendering import normalize_depth_map
import text3d2video.rerun_util as ru
import torchvision.transforms.functional as TF
import faiss
from PIL import Image
from text3d2video.sd_feature_extraction import SDFeatureExtractor
from text3d2video.util import project_vertices_to_features, multiview_cameras, random_solid_color_img
from text3d2video.visualization import RgbPcaUtil
from pytorch3d.renderer import FoVPerspectiveCameras

from text3d2video.wandb_util import AnimationArtifact, MVFeaturesArtifact, first_used_artifact_of_type


def aggregate_3d_features(
        cameras: FoVPerspectiveCameras,
        feature_maps: list[torch.Tensor],
        mesh: Meshes,
        log_pca_features: bool = True
) -> torch.Tensor:

    rr.log('mesh', ru.pt3d_mesh(mesh))

    if log_pca_features:

        # take first channel as preiminary visualization
        rgb_feature_maps = [f[0:3] for f in feature_maps]

        # log PCA feature maps
        n_views = len(cameras)
        for i in range(n_views):
            feature_map = rgb_feature_maps[i]
            res = feature_map.shape[1]
            ru.log_pt3d_FovCamrea(f'cam_{i}', cameras, batch_idx=i, res=res)
            rr.log(f'cam_{i}', rr.Image(TF.to_pil_image(feature_map)))

    # initialize empty D-dimensional vertex features
    feature_dim = feature_maps[0].shape[0]
    vertex_features = torch.zeros(mesh.num_verts_per_mesh()[0], feature_dim)
    vertex_feature_count = torch.zeros(mesh.num_verts_per_mesh()[0])

    n_views = len(cameras)

    for i in range(n_views):
        # project view features to vertices
        feature_map = feature_maps[i]
        view_vertex_features = project_vertices_to_features(
            mesh,
            cameras,
            feature_map,
            batch_idx=i
        ).cpu()

        # indices of vertices with nonzero view features
        nonzero_indices = torch.where(
            torch.any(view_vertex_features != 0, dim=1))[0]

        # update vertex features
        vertex_features += view_vertex_features
        vertex_feature_count[nonzero_indices] += 1

        rr.log('mesh', ru.pt3d_mesh(
            mesh, vertex_colors=vertex_features[:, 0:3]))

    return vertex_features


if __name__ == "__main__":

    features_art = 'multiview_features:latest'

    wandb.init(project="diffusion-3d-features",
               job_type='aggregate_3d_features')

    # init 3D logging
    rr.init('multiview_features')
    rr.serve()
    ru.pt3d_setup()

    # log blueprint
    blueprint = rrb.Blueprint(rrb.Spatial3DView())
    rr.send_blueprint(blueprint)

    # download multiview features
    mv_features_artifact = wandb.use_artifact(features_art)
    mv_features_data = MVFeaturesArtifact(mv_features_artifact)
    cameras = mv_features_data.get_cameras()
    features = mv_features_data.get_features()

    # get mesh
    mv_features_run = mv_features_artifact.logged_by()
    anim_artifact = first_used_artifact_of_type(
        mv_features_run, AnimationArtifact.type)
    anim_artifact_data = AnimationArtifact(anim_artifact)
    mesh_path = anim_artifact_data.get_mesh_path()

    # load mesh
    device = 'cuda:0'
    mesh: Meshes = load_objs_as_meshes([mesh_path], device=device)

    vertex_features = aggregate_3d_features(cameras, features, mesh)

    pca = RgbPcaUtil(vertex_features.shape[1])
    pca.fit(vertex_features)
    vertex_features_rgb = pca.features_to_rgb(vertex_features)

    print(vertex_features.shape)
