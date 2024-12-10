from dataclasses import dataclass
from typing import Dict, List

import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from torch import Tensor

from text3d2video.artifacts.diffusion_data import (
    AttnFeaturesWriter,
    DiffusionData,
    DiffusionDataCfg,
    DiffusionDataWriter,
    LatentsWriter,
)
from text3d2video.h5_util import read_tensor_from_dataset, write_tensor_as_dataset
from text3d2video.util import assert_tensor_shape
from text3d2video.wandb_util import ArtifactWrapper


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
    """

    save_kf_post_attn: bool
    save_aggregated_features: bool

    def __init__(
        self,
        diff_data,
        save_kf_post_attn=True,
        save_aggregated_features=True,
        data_path="gr_data",
    ):
        super().__init__(diff_data, data_path)
        self.save_kf_post_attn = save_kf_post_attn
        self.save_aggregated_features = save_aggregated_features

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

    # writing

    def write_kf_indices(self, t: int, kf_indices: Tensor):
        path = self._kf_indices_path(t)
        self.write_tensor(path, kf_indices, timestep=t)

    def write_vertex_features(self, t: int, vert_features: Dict[str, Tensor]):
        for layer, features in vert_features.items():
            if self.save_aggregated_features:
                path = self._vert_features_path(t, layer)
                self.write_tensor(path, features, timestep=t, attn_path=layer)

    def write_kf_post_attn(self, t: int, post_attn_features: Dict[str, Tensor]):
        for layer, features in post_attn_features.items():
            if self.save_kf_post_attn:
                path = self._kf_features_path(t, layer)
                self.write_tensor(path, features, timestep=t, attn_path=layer)

    def write_post_attn_pre_injection(
        self, t: int, layer: str, feature_maps, chunk_indices: Tensor
    ):
        assert_tensor_shape(feature_maps, ("B", "T", "d", "H", "W"))

        for frame_i in self.diff_data.save_frame_indices:
            idx_in_chunk = (chunk_indices == frame_i).nonzero(as_tuple=True)[0]
            if len(idx_in_chunk) == 0:
                continue
            idx_in_chunk = int(idx_in_chunk)

            feature_map = feature_maps[0, idx_in_chunk, :, :, :]
            path = self._post_attn_pre_injection_path(t, frame_i, layer)
            self.write_tensor(
                path, feature_map, timestep=t, frame_i=frame_i, attn_path=layer
            )

    def write_post_attn_post_injection(
        self, t: int, layer: str, feature_maps, chunk_indices: Tensor
    ):
        assert_tensor_shape(feature_maps, ("B", "T", "d", "H", "W"))

        for frame_i in self.diff_data.save_frame_indices:
            idx_in_chunk = (chunk_indices == frame_i).nonzero(as_tuple=True)[0]
            if len(idx_in_chunk) == 0:
                continue
            idx_in_chunk = int(idx_in_chunk)

            feature_map = feature_maps[0, idx_in_chunk, :, :, :]
            path = self._post_attn_post_injection_path(t, frame_i, layer)
            self.write_tensor(
                path, feature_map, timestep=t, frame_i=frame_i, attn_path=layer
            )

    # reading

    def read_kf_post_attn(self, t: int, layer: str) -> Dict[str, Tensor]:
        path = self._kf_features_path(t, layer)
        return self.read_tensor(path)

    def read_kf_indices(self, t: int) -> Tensor:
        path = self._kf_indices_path(t)
        return self.read_tensor(path)

    def read_vertex_features(self, t: int, layer: str) -> Dict[str, Tensor]:
        path = self._vert_features_path(t, layer)
        return self.read_tensor(path)

    def read_post_attn_pre_injection(self, t: int, frame_i: int, layer: str):
        path = self._post_attn_pre_injection_path(t, frame_i, layer)
        return self.read_tensor(path)

    def read_post_attn_post_injection(self, t: int, frame_i: int, layer: str):
        path = self._post_attn_post_injection_path(t, frame_i, layer)
        return self.read_tensor(path)


class GrDataArtifact(ArtifactWrapper):
    wandb_artifact_type = "gr_data"

    # config for saving
    config: GrSaveConfig
    diffusion_data: DiffusionData
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
        # set up diffusion data object
        diffusion_data_cfg = DiffusionDataCfg(
            enabled=config.enabled,
            n_save_steps=config.n_timesteps,
            n_save_frames=config.n_frames,
            attn_paths=config.module_paths,
        )

        art.diffusion_data = DiffusionData(diffusion_data_cfg, art.h5_file_path())
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
        self.diffusion_data.calculate_save_steps(scheduler)
        self.diffusion_data.calculate_save_frames(n_frames)
        self.diffusion_data.begin_recording()

    def end_recording(self):
        self.diffusion_data.end_recording()

    # Reading methods

    def read_latent(self, t: int, frame_i: int) -> Tensor:
        return self.latents_writer.read_latent(t, frame_i)

    def read_latents_at_frame(self, frame_i: int) -> Tensor:
        times = self.diffusion_data.save_times
        return torch.stack([self.read_latent(t, frame_i) for t in times])

    def read_latents_at_timestep(self, t: int) -> Tensor:
        frame_indices = self.diffusion_data.save_frame_indices
        return torch.stack([self.read_latent(t, frame_i) for frame_i in frame_indices])
