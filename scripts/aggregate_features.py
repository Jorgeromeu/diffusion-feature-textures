from enum import Enum
import math
from pathlib import Path
from typing import List

import hydra
import rerun as rr
import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from omegaconf import DictConfig
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Meshes
from tqdm import tqdm

from text3d2video.disk_multidict import TensorDiskMultiDict
import text3d2video.rerun_util as ru
import text3d2video.wandb_util as wu
import wandb
from text3d2video.artifacts.multiview_features_artifact import MVFeaturesArtifact
from text3d2video.artifacts.vertex_atributes_artifact import VertAttributesArtifact
from text3d2video.util import project_vertices_to_features
from text3d2video.visualization import RgbPcaUtil


class AggregationType(Enum):
    MEAN = 0
    FIRST = 1


def aggregate_3d_features(
    cameras: FoVPerspectiveCameras,
    feature_maps: List[torch.Tensor],
    mesh: Meshes,
    log_pca_features: bool = True,
    aggregation_type: int = AggregationType.MEAN,
    interpolation_mode: str = "bilinear",
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
            mesh, cameras, feature_map, batch_idx=view_i, mode=interpolation_mode
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


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run(cfg: DictConfig):

    aggr_cfg = cfg.aggregate_3d_features

    # init run
    wu.init_run(job_type="aggregate_3d_features", dev_run=cfg.dev_run)
    wandb.config.update(dict(aggr_cfg))

    # init 3D logging
    ru.set_logging_state(cfg.rerun_enabled)
    rr.init("multiview_features")
    rr.serve()
    ru.pt3d_setup()

    # get multiview features artifact
    mv_features = wu.get_artifact(aggr_cfg.mv_features_artifact_tag)
    mv_features = MVFeaturesArtifact.from_wandb_artifact(mv_features)

    # recover unposed mesh from lineage
    animation = mv_features.get_animation_from_lineage()
    mesh = animation.load_static_mesh("cuda:0")

    # get views
    cameras = mv_features.get_cameras()

    # store all aggregated 3D features
    vertex_features_path = Path("outs/vertex_features")
    all_vertex_features = TensorDiskMultiDict(vertex_features_path, init_empty=True)

    # iterate over saved layers and timesteps
    layers = mv_features.get_key_values("layer")
    timesteps = mv_features.get_key_values("timestep")

    for layer in tqdm(layers, "layers"):
        for timestep in tqdm(timesteps, "timesteps"):

            # get feature map per view
            feature_identifier = {"layer": layer, "timestep": timestep}
            features = [
                mv_features.get_feature(i, feature_identifier)
                for i in mv_features.view_indices()
            ]

            # reshape features to square
            feature_maps = []
            for feature in features:
                seq_len, _ = feature.shape
                feature_map_size = int(math.sqrt(seq_len))
                feature_map = rearrange(feature, "(h w) d -> d h w", h=feature_map_size)
                feature_maps.append(feature_map)

            # aggregate features to 3D
            aggregation_type = AggregationType[str(aggr_cfg.aggregation_method).upper()]
            vertex_features = aggregate_3d_features(
                cameras,
                feature_maps,
                mesh,
                aggregation_type=aggregation_type,
                interpolation_mode=aggr_cfg.projection_interp_method,
                log_pca_features=cfg.rerun_enabled,
            )

            # save 3D features
            all_vertex_features.add(feature_identifier, vertex_features)

    # save features as artifact
    artifact = VertAttributesArtifact.create_wandb_artifact(
        aggr_cfg.out_artifact_name,
        features_path=all_vertex_features.path.absolute(),
    )

    wu.log_artifact_if_enabled(artifact)
    wandb.finish()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run()
