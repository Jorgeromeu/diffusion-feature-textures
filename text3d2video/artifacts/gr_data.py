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
from text3d2video.wandb_util import ArtifactWrapper


@dataclass
class GrSaveConfig:
    enabled: bool
    n_frames: int
    n_timesteps: int
    save_latents: bool
    save_q: bool
    save_k: bool
    save_v: bool
    out_artifact: str
    module_paths: list[str]


class GrDataWriter(DiffusionDataWriter):
    """
    Class to write attention data
    """

    def __init__(self, diff_data, data_path="gr_data"):
        super().__init__(diff_data, data_path)

    def _layer_path(self, t: int, layer: str):
        return f"{self.data_path}/time_{t}/{layer}"

    def _vert_features_path(self, t: int, layer: str):
        return f"{self._layer_path(t, layer)}/vert_features"

    def _kf_features_path(self, t: int, layer: str):
        return f"{self._layer_path(t, layer)}/kf_spatial_features"

    def _rendered_feature_path(self, t: int, frame_i: int, layer: str):
        return f"{self._layer_path(t, layer)}/frame_{frame_i}/rendered"

    # writing

    def write_vertex_features(self, t: int, vert_features: Dict[str, Tensor]):
        for layer, features in vert_features.items():
            path = self._vert_features_path(t, layer)
            write_tensor_as_dataset(self.diff_data.h5_write_fp, path, features)

    def write_kf_post_attn(self, t: int, post_attn_features: Dict[str, Tensor]):
        for layer, features in post_attn_features.items():
            path = self._kf_features_path(t, layer)
            write_tensor_as_dataset(self.diff_data.h5_write_fp, path, features)

    def write_rendered_post_attn(
        self, t: int, frame_indices: List[int], rendered_features: Dict[str, Tensor]
    ):
        pass

    # reading

    def read_kf_post_attn(self, t: int, layer: str) -> Dict[str, Tensor]:
        path = self._kf_features_path(t, layer)
        return read_tensor_from_dataset(self.diff_data.h5_file_path, path)

    def read_vertex_features(self, t: int, layer: str) -> Dict[str, Tensor]:
        path = self._vert_features_path(t, layer)
        print(path)
        vert_features = read_tensor_from_dataset(self.diff_data.h5_file_path, path)
        return vert_features


class GrDataArtifact(ArtifactWrapper):
    wandb_artifact_type = "gr_data"

    # config for saving
    config: GrSaveConfig
    diffusion_data: DiffusionData
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
        art.latents_writer = LatentsWriter(art.diffusion_data)
        art.attn_writer = AttnFeaturesWriter(
            art.diffusion_data,
            save_q=config.save_q,
            save_k=config.save_k,
            save_v=config.save_v,
        )

        art.gr_writer = GrDataWriter(art.diffusion_data)
        return art

    def begin_recording(self, scheduler: SchedulerMixin, n_frames: int):
        self.diffusion_data.calculate_save_steps(scheduler)
        self.diffusion_data.calculate_save_frames(n_frames)
        self.diffusion_data.begin_recording()

    def end_recording(self):
        self.diffusion_data.end_recording()

    def save_latents(self, t: int, latents: Tensor):
        if self.config.save_latents:
            self.latents_writer.write_latents_batched(t, latents)

    def save_qkv(self, t: int, q: Tensor, k: Tensor, v: Tensor, layer: str):
        self.attn_writer.write_qkv_batched(t, layer, q, k, v)

    # Reading methods

    def read_latent(self, t: int, frame_i: int) -> Tensor:
        return self.latents_writer.read_latent(t, frame_i)

    def read_latents_at_frame(self, frame_i: int) -> Tensor:
        times = self.diffusion_data.save_times
        return torch.stack([self.read_latent(t, frame_i) for t in times])

    def read_latents_at_timestep(self, t: int) -> Tensor:
        frame_indices = self.diffusion_data.save_frame_indices
        return torch.stack([self.read_latent(t, frame_i) for frame_i in frame_indices])
