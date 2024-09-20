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


class AggregationType:
    MEAN = 0
    FIRST = 1


def aggregate_3d_features(
        cameras: FoVPerspectiveCameras,
        feature_maps: list[torch.Tensor],
        mesh: Meshes,
        log_pca_features: bool = True,
        aggregation_type: int = AggregationType.MEAN
) -> torch.Tensor:

    rr_seq = ru.TimeSequence("steps")
    rr.log('mesh', ru.pt3d_mesh(mesh))

    if log_pca_features:
        # fit PCA matrix
        feature_dim = feature_maps[0].shape[0]
        pca = RgbPcaUtil(feature_dim)
        features_all = torch.stack(feature_maps)
        features_all = rearrange(features_all, 'v c h w -> (v h w) c')
        pca.fit(features_all)

        # PCA the feature maps
        rgb_feature_maps = [pca.feature_map_to_rgb(f) for f in feature_maps]

    # initialize empty D-dimensional vertex features
    feature_dim = feature_maps[0].shape[0]
    vert_features = torch.zeros(mesh.num_verts_per_mesh()[0], feature_dim)
    vert_feature_cnt = torch.zeros(mesh.num_verts_per_mesh()[0])

    # for each view
    for view_i, _ in enumerate(cameras):

        rr_seq.step()

        # project view features to vertices
        feature_map = feature_maps[view_i]
        view_vert_features = project_vertices_to_features(
            mesh,
            cameras,
            feature_map,
            batch_idx=view_i,
            mode='bilinear'
        ).cpu()

        if aggregation_type == AggregationType.FIRST:
            # update empty entries in vert_features
            empty_vert_idxs = vert_feature_cnt == 0
            vert_features[empty_vert_idxs] = view_vert_features[empty_vert_idxs]
            aggr_vert_features = vert_features

        # keep track of number of features per vertex
        nonzero_view_idxs = torch.where(
            torch.any(view_vert_features != 0, dim=1))[0]
        vert_feature_cnt[nonzero_view_idxs] += 1

        if aggregation_type == AggregationType.MEAN:
            # update vertex features

            # get indices of vertices with nonzero features
            vert_features += view_vert_features
            aggr_vert_features = vert_features / \
                torch.clamp(vert_feature_cnt, min=1).unsqueeze(1)

        if log_pca_features:

            res = rgb_feature_maps[view_i].shape[-1]
            ru.log_pt3d_FovCamrea(
                f'cam_{view_i}', cameras, batch_idx=view_i, res=res)
            rr.log(f'cam_{view_i}', rr.Image(
                TF.to_pil_image(rgb_feature_maps[view_i])))

            vert_features_rgb = pca.features_to_rgb(aggr_vert_features)
            rr.log('mesh', ru.pt3d_mesh(mesh, vertex_colors=vert_features_rgb))

    if log_pca_features:
        # recompute PCA on final features
        rr_seq.step()
        rgb_vert_features = pca.fit(aggr_vert_features)
        rr.log('mesh', ru.pt3d_mesh(mesh, vertex_colors=rgb_vert_features))

    return aggr_vert_features


if __name__ == "__main__":

    features_art = 'fox_features:latest'
    feature_identifier = {'layer': 'level_2', 'timestep': '20'}
    aggregation_type = AggregationType.MEAN

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
    features = [mv_features_data.get_feature(i, feature_identifier) for i in
                mv_features_data.view_indices()]

    # get mesh
    mv_features_run = mv_features_artifact.logged_by()
    anim_artifact = first_used_artifact_of_type(
        mv_features_run, AnimationArtifact.type)
    anim_artifact_data = AnimationArtifact(anim_artifact)
    mesh_path = anim_artifact_data.get_mesh_path()

    # load mesh
    device = 'cuda:0'
    mesh: Meshes = load_objs_as_meshes([mesh_path], device=device)

    vertex_features = aggregate_3d_features(
        cameras,
        features,
        mesh,
        aggregation_type=aggregation_type
    )

    pca = RgbPcaUtil(vertex_features.shape[1])
    pca.fit(vertex_features)
    vertex_features_rgb = pca.features_to_rgb(vertex_features)

    print(vertex_features.shape)
