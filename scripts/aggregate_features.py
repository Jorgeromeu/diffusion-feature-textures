from typing import List

import rerun as rr
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Meshes

import text3d2video.rerun_util as ru
import wandb
from text3d2video.artifacts.animation_artifact import AnimationArtifact
from text3d2video.artifacts.multiview_features_artifact import MVFeaturesArtifact
from text3d2video.util import project_vertices_to_features
from text3d2video.visualization import RgbPcaUtil
from text3d2video.wandb_util import first_used_artifact_of_type


class AggregationType:
    MEAN = 0
    FIRST = 1


def aggregate_3d_features(
    cameras: FoVPerspectiveCameras,
    feature_maps: List[torch.Tensor],
    mesh: Meshes,
    log_pca_features: bool = True,
    aggregation_type: int = AggregationType.MEAN,
) -> torch.Tensor:

    rr_seq = ru.TimeSequence("steps")
    rr.log("mesh", ru.pt3d_mesh(mesh))

    if log_pca_features:
        # fit PCA matrix
        feature_dim = feature_maps[0].shape[0]
        pca = RgbPcaUtil(feature_dim)
        features_all = torch.stack(feature_maps)
        features_all = rearrange(features_all, "v c h w -> (v h w) c")
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
            mesh, cameras, feature_map, batch_idx=view_i, mode="bilinear"
        ).cpu()

        if aggregation_type == AggregationType.FIRST:
            # update empty entries in vert_features
            empty_vert_idxs = vert_feature_cnt == 0
            vert_features[empty_vert_idxs] = view_vert_features[empty_vert_idxs]
            aggr_vert_features = vert_features

        # keep track of number of features per vertex
        nonzero_view_idxs = torch.where(torch.any(view_vert_features != 0, dim=1))[0]
        vert_feature_cnt[nonzero_view_idxs] += 1

        if aggregation_type == AggregationType.MEAN:
            # update vertex features

            # get indices of vertices with nonzero features
            vert_features += view_vert_features
            aggr_vert_features = vert_features / torch.clamp(
                vert_feature_cnt, min=1
            ).unsqueeze(1)

        if log_pca_features:

            res = rgb_feature_maps[view_i].shape[-1]
            ru.log_pt3d_FovCamrea(f"cam_{view_i}", cameras, batch_idx=view_i, res=res)
            rr.log(f"cam_{view_i}", rr.Image(TF.to_pil_image(rgb_feature_maps[view_i])))

            vert_features_rgb = pca.features_to_rgb(aggr_vert_features)
            rr.log("mesh", ru.pt3d_mesh(mesh, vertex_colors=vert_features_rgb))

    if log_pca_features:
        # recompute PCA on final features
        rr_seq.step()
        rgb_vert_features = pca.fit(aggr_vert_features)
        rr.log("mesh", ru.pt3d_mesh(mesh, vertex_colors=rgb_vert_features))

    return aggr_vert_features


if __name__ == "__main__":

    features_art = "fox_features:latest"
    feature_identifier = {"layer": "level_2", "timestep": "20"}
    aggregation_type = AggregationType.MEAN

    wandb.init(project="diffusion-3d-features", job_type="aggregate_3d_features")

    # init 3D logging
    rr.init("multiview_features")
    rr.serve()
    ru.pt3d_setup()

    # download multiview features artifact
    mv_features_artifact = wandb.use_artifact(features_art)
    mv_features_artifact = MVFeaturesArtifact.from_wandb_artifact(mv_features_artifact)

    # get feature map per camera
    cameras = mv_features_artifact.get_cameras()
    features = [
        mv_features_artifact.get_feature(i, feature_identifier)
        for i in mv_features_artifact.view_indices()
    ]

    # get anim artifact
    mv_features_run = mv_features_artifact.logged_by()
    anim_artifact = first_used_artifact_of_type(
        mv_features_run, AnimationArtifact.artifact_type
    )
    anim_artifact = AnimationArtifact.from_wandb_artifact(anim_artifact)

    # get mesh
    device = "cuda:0"
    mesh = anim_artifact.load_static_mesh(device)

    # aggregate features to 3D
    vertex_features = aggregate_3d_features(
        cameras, features, mesh, aggregation_type=aggregation_type
    )

    # PCA the feature maps
    pca = RgbPcaUtil(vertex_features.shape[1])
    pca.fit(vertex_features)
    vertex_features_rgb = pca.features_to_rgb(vertex_features)
