import math
from enum import Enum

import hydra
import rerun as rr
import torch
from einops import rearrange
from jaxtyping import Float
from omegaconf import DictConfig
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Meshes
from tqdm import tqdm

import text3d2video.rerun_util as ru
import text3d2video.wandb_util as wu
from text3d2video.artifacts.multiview_features_artifact import MVFeaturesArtifact
from text3d2video.artifacts.vertex_atributes_artifact import VertAttributesArtifact
from text3d2video.util import project_vertices_to_features


class AggregationType(Enum):
    MEAN = 0
    FIRST = 1


def aggregate_3d_features(
    cameras: FoVPerspectiveCameras,
    meshes: Meshes,
    feature_maps: Float[torch.Tensor, "n c h w"],
    aggregation_type: int = AggregationType.MEAN,
    interpolation_mode: str = "bilinear",
) -> Float[torch.Tensor, "v c"]:
    assert len(feature_maps) == len(cameras) == len(meshes)

    # initialize empty D-dimensional vertex features
    feature_dim = feature_maps[0].shape[0]
    vert_features = torch.zeros(meshes.num_verts_per_mesh()[0], feature_dim)
    vert_feature_cnt = torch.zeros(meshes.num_verts_per_mesh()[0])

    # for each view
    for i, _ in enumerate(cameras):
        feature_map = feature_maps[i]
        camera = cameras[i]
        mesh = meshes[i]

        # project view features to vertices
        view_vert_features = project_vertices_to_features(
            mesh, camera, feature_map, mode=interpolation_mode
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
            # get indices of vertices with nonzero features
            vert_features += view_vert_features
            aggr_vert_features = vert_features / torch.clamp(
                vert_feature_cnt, min=1
            ).unsqueeze(1)

    return aggr_vert_features


@hydra.main(version_base=None, config_path="../config", config_name="config")
def run(cfg: DictConfig):
    # init run
    do_run = wu.setup_run(cfg)
    if not do_run:
        return

    aggr_cfg = cfg.aggregate_3d_features

    # init 3D logging
    ru.set_logging_state(cfg.rerun_enabled)
    rr.init("multiview_features")
    rr.serve()
    ru.pt3d_setup()

    # get multiview features artifact
    mv_features = MVFeaturesArtifact.from_wandb_artifact_tag(
        aggr_cfg.mv_features_artifact_tag, download=True
    )

    # get original mesh from lineage
    animation = mv_features.get_animation_from_lineage()
    mesh = animation.load_unposed_mesh("cuda:0")

    # get views
    cameras = mv_features.get_cameras()

    # create empty out artifact
    out_artifact = VertAttributesArtifact.create_empty_artifact(
        aggr_cfg.out_artifact_name
    )

    # store all aggregated 3D features
    all_vertex_features = out_artifact.get_features_disk_dict()

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
            )

            # save 3D features
            all_vertex_features.add(feature_identifier, vertex_features)

    # save features as artifact
    out_artifact.log_if_enabled()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run()
