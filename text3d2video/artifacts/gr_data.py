import pickle
from dataclasses import dataclass
from typing import Dict

from diffusers.schedulers.scheduling_utils import SchedulerMixin
from torch import Tensor

from text3d2video.artifacts.diffusion_data import (
    AttnFeaturesWriter,
    DiffusionDataLogger,
    DiffusionDataWriter,
    LatentsWriter,
)
from text3d2video.util import assert_tensor_shapes
from wandb_util.wandb_util import ArtifactWrapper


@dataclass
class GrSaveConfig:
    enabled: bool
    n_frames: int
    n_timesteps: int
    out_artifact: str
    module_paths: list[str]
    save_latents: bool
    save_q: bool
    save_k: bool
    save_v: bool
    save_kf_post_attn: bool
    save_aggregated_features: bool
    save_feature_images: bool


class GrDataWriter(DiffusionDataWriter):
    """
    Class to write attention data
    time_t/
        kf_indices
        layer/
            kf_spatial_features
            vert_features
            frame_i/
                pre_injection
                post_injection
                rendered
    """

    save_kf_post_attn: bool
    save_aggregated_features: bool
    save_feature_images: bool

    def __init__(
        self,
        diff_data,
        save_kf_post_attn=True,
        save_aggregated_features=True,
        save_feature_images=True,
        data_path="gr_data",
    ):
        super().__init__(diff_data, data_path)
        self.save_kf_post_attn = save_kf_post_attn
        self.save_aggregated_features = save_aggregated_features
        self.save_feature_images = save_feature_images

    def _time_path(self, t: int):
        return f"{self.data_path}/time_{t}"

    def _kf_indices_path(self, t: int):
        return f"{self._time_path(t)}/kf_indices"

    def _layer_path(self, t: int, layer: str):
        return f"{self._time_path(t)}/{layer}"

    def _vert_features_path(self, t: int, layer: str):
        return f"{self._layer_path(t, layer)}/vert_features"

    def _kf_features_path(self, t: int, layer: str):
        return f"{self._layer_path(t, layer)}/kf_spatial_features"

    def _post_attn_pre_injection_path(self, t, frame_i, layer: str):
        return f"{self._layer_path(t, layer)}/frame_{frame_i}/pre_injection"

    def _post_attn_post_injection_path(self, t, frame_i, layer: str):
        return f"{self._layer_path(t, layer)}/frame_{frame_i}/post_injection"

    def _post_attn_render_path(self, t, frame_i, layer: str):
        return f"{self._layer_path(t, layer)}/frame_{frame_i}/render"

    # writing

    def write_kf_indices(self, t: int, kf_indices: Tensor):
        path = self._kf_indices_path(t)
        self.write_tensor(path, kf_indices, timestep=t)

    def write_vertex_features(self, t: int, vert_features: Dict[str, Tensor]):
        if not self.save_aggregated_features:
            return
        for layer, features in vert_features.items():
            path = self._vert_features_path(t, layer)
            self.write_tensor(path, features, timestep=t, attn_path=layer)

    def write_kf_post_attn(self, t: int, post_attn_features: Dict[str, Tensor]):
        if not self.save_kf_post_attn:
            return
        for layer, features in post_attn_features.items():
            path = self._kf_features_path(t, layer)
            self.write_tensor(path, features, timestep=t, attn_path=layer)

    def write_post_attn_pre_injection(
        self, t: int, layer: str, feature_maps, chunk_indices: Tensor
    ):
        assert_tensor_shapes(
            [(feature_maps, ("B", "T", "d", "H", "W")), (chunk_indices, ("T"))]
        )

        if not self.save_feature_images:
            return

        for tensor_idx, frame_i in enumerate(chunk_indices):
            feature_map = feature_maps[0, tensor_idx, :, :, :]
            path = self._post_attn_post_injection_path(t, frame_i, layer)
            path = self._post_attn_pre_injection_path(t, frame_i, layer)
            self.write_tensor(
                path, feature_map, timestep=t, frame_i=frame_i, attn_path=layer
            )

    def write_post_attn_post_injection(
        self, t: int, layer: str, feature_maps, chunk_indices: Tensor
    ):
        assert_tensor_shapes(
            [(feature_maps, ("B", "T", "d", "H", "W")), (chunk_indices, ("T"))]
        )

        if not self.save_feature_images:
            return

        for tensor_idx, frame_i in enumerate(chunk_indices):
            feature_map = feature_maps[0, tensor_idx, :, :, :]
            path = self._post_attn_post_injection_path(t, frame_i, layer)
            self.write_tensor(
                path, feature_map, timestep=t, frame_i=frame_i, attn_path=layer
            )

    def write_post_attn_renders(
        self, t: int, layer: str, feature_maps, chunk_indices: Tensor
    ):
        assert_tensor_shapes(
            [(feature_maps, ("B", "T", "d", "H", "W")), (chunk_indices, ("T"))]
        )

        if not self.save_feature_images:
            return

        for tensor_idx, frame_i in enumerate(chunk_indices):
            feature_map = feature_maps[0, tensor_idx, :, :, :]
            path = self._post_attn_render_path(t, frame_i, layer)
            self.write_tensor(
                path, feature_map, timestep=t, frame_i=frame_i, attn_path=layer
            )

    # reading

    def read_kf_post_attn(self, t: int, layer: str) -> Dict[str, Tensor]:
        path = self._kf_features_path(t, layer)
        return self._read_tensor(path)

    def read_kf_indices(self, t: int) -> Tensor:
        path = self._kf_indices_path(t)
        return self._read_tensor(path)

    def read_vertex_features(self, t: int, layer: str) -> Dict[str, Tensor]:
        path = self._vert_features_path(t, layer)
        return self._read_tensor(path)

    def read_post_attn_pre_injection(self, t: int, frame_i: int, layer: str):
        path = self._post_attn_pre_injection_path(t, frame_i, layer)
        return self._read_tensor(path)

    def read_post_attn_post_injection(self, t: int, frame_i: int, layer: str):
        path = self._post_attn_post_injection_path(t, frame_i, layer)
        return self._read_tensor(path)

    def read_post_attn_render(self, t: int, frame_i: int, layer: str):
        path = self._post_attn_render_path(t, frame_i, layer)
        return self._read_tensor(path)


class GrDataArtifact(ArtifactWrapper):
    wandb_artifact_type = "gr_data"

    # config for saving
    config: GrSaveConfig
    diffusion_data: DiffusionDataLogger
    # diffusion data writers
    attn_writer: AttnFeaturesWriter
    latents_writer: LatentsWriter
    gr_writer: GrDataWriter

    def h5_file_path(self):
        return self.folder / "data.h5"

    @classmethod
    def init_from_config(cls, config: GrSaveConfig):
        art = GrDataArtifact.create_empty_artifact(config.out_artifact)
        art.config = config

        # diffusion data
        art.diffusion_data = DiffusionDataLogger(
            art.h5_file_path(),
            enabled=config.enabled,
            path_greenlist=config.module_paths,
        )

        # writers
        art.latents_writer = LatentsWriter(
            art.diffusion_data, enabled=config.save_latents
        )
        art.attn_writer = AttnFeaturesWriter(
            art.diffusion_data,
            save_q=config.save_q,
            save_k=config.save_k,
            save_v=config.save_v,
        )

        art.gr_writer = GrDataWriter(
            art.diffusion_data,
            save_kf_post_attn=config.save_kf_post_attn,
            save_aggregated_features=config.save_aggregated_features,
        )
        return art

    def begin_recording(self, scheduler: SchedulerMixin, n_frames: int):
        self.diffusion_data.calc_evenly_spaced_noise_noise_levels(scheduler, 5)
        self.diffusion_data.calc_evenly_spaced_frame_indices(n_frames, 5)
        self.diffusion_data.begin_recording()

    def end_recording(self):
        self.diffusion_data.end_recording()

    # Reading methods
