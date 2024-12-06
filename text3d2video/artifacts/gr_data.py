from dataclasses import dataclass

import h5py
import numpy as np
import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from torch import Tensor

from text3d2video.artifacts.tensors_artifact import H5Artifact
from text3d2video.util import ordered_sample, ordered_sample_indices


@dataclass
class GrSaveConfig:
    enabled: bool
    save_latents: bool
    save_q: bool
    save_k: bool
    save_v: bool
    save_features: bool
    save_features_3d: bool
    n_frames: int
    n_timesteps: int
    out_artifact: str
    module_paths: list[str]


class GrDataArtifact(H5Artifact):
    wandb_artifact_type = "gr_data"

    # config for saving
    config: GrSaveConfig
    save_frame_indices: list[int]
    save_timesteps: list[int]

    def _latent_h5_path(self, t: int, frame_i: int):
        return f"time_{t}/frame_{frame_i}/latents"

    def _feature_h5_path(
        self, t: int, frame_i: int, module_path: str, feature_name: str
    ):
        return f"time_{t}/frame_{frame_i}/layer_{module_path}/{feature_name}"

    def _3d_features_path(self, t: int, module_path: str):
        return f"time_{t}/layer_{module_path}/vertex_features"

    def should_save_timestep(self, t: int):
        if self.save_timesteps == []:
            return True

        return t in self.save_timesteps

    def should_save_frame(self, frame_i: int):
        if self.save_frame_indices == []:
            return True

        return frame_i in self.save_frame_indices

    # Initialization Methods

    @classmethod
    def init_from_cfg(cls, cfg: GrSaveConfig):
        """
        Create a GrDataArtifact from a GrSaveConfig
        :param cfg: GrSaveConfig
        """

        art = cls.create_empty_artifact(cfg.out_artifact)
        art.config = cfg
        art.save_frame_indices = []
        art.save_timesteps = []
        return art

    def compute_save_frame_indices(self, n_frames: int):
        frame_indices = list(range(n_frames))
        self.save_frame_indices = ordered_sample_indices(
            frame_indices, self.config.n_frames
        )

    def compute_save_timesteps(self, scheduler: SchedulerMixin):
        diffusion_steps = torch.cat([scheduler.timesteps, torch.Tensor([0])])
        self.save_timesteps = ordered_sample(diffusion_steps, self.config.n_timesteps)

    def begin_recording(self):
        self.open_h5_file()
        self.create_dataset("frame_indices", Tensor(self.save_frame_indices))
        self.create_dataset("timesteps", Tensor(self.save_timesteps))

    def end_recording(self):
        self.close_h5_file()

    # Writing methods

    def save_latents(self, latents: Tensor, t: int):
        save_latents = (
            self.config.enabled
            and self.config.save_latents
            and t in self.save_timesteps
        )

        if not save_latents:
            return

        for frame_i in self.save_frame_indices:
            latent = latents[frame_i]
            self.create_dataset(
                self._latent_h5_path(t, frame_i),
                latent.cpu(),
                dim_names=["channel", "height", "width"],
            )

    def should_save_feature(self, t: int, module_path: str):
        return (
            self.config.enabled
            and t in self.save_timesteps
            and module_path in self.config.module_paths
        )

    def save_qkv(
        self, qrys: Tensor, keys: Tensor, values: Tensor, t: int, module_path: str
    ):
        if not self.should_save_feature(t, module_path):
            return

        def save_feature(name, feature):
            for frame_i in self.save_frame_indices:
                self.create_dataset(
                    self._feature_h5_path(t, frame_i, module_path, name),
                    feature[frame_i].cpu(),
                    dim_names=["sequence", "channel"],
                )

        if self.config.save_q:
            save_feature("query", qrys)
        if self.config.save_k:
            save_feature("key", keys)
        if self.config.save_v:
            save_feature("value", values)

    def _save_attn_out(self, attn_out, t, module_path, attn_out_name):
        if not self.should_save_feature(t, module_path):
            return

        for frame_i in self.save_frame_indices:
            self.create_dataset(
                self._feature_h5_path(t, frame_i, module_path, attn_out_name),
                attn_out[0, frame_i].cpu(),
                dim_names=["channel", "height", "width"],
            )

    def save_attn_out_pre(self, attn_out, t, module_path):
        self._save_attn_out(attn_out, t, module_path, "attn_out_pre")

    def save_attn_out_post(self, attn_out, t, module_path):
        self._save_attn_out(attn_out, t, module_path, "attn_out_post")

    def save_vertex_features(self, all_vertex_features: dict[str], t: int):
        for module, (vert_features, res) in all_vertex_features.items():
            pass

    # Reading methods

    def read_frame_indices(self):
        indices = self.read_dataset_np("frame_indices").astype(int)
        return indices.tolist()

    def read_timesteps(self):
        timesteps = self.read_dataset_np("timesteps").astype(int)
        return timesteps.tolist()

    def read_module_paths(self):
        modules = self.read_dataset_np("modules")
        modules = [m.decode() for m in modules]
        return modules

    def read_latent(self, t: int, frame_i: int) -> Tensor:
        path = self._latent_h5_path(t, frame_i)
        return self.read_dataset_np(path)

    def read_feature(
        self, t: int, frame_i: int, module_path: str, feature_name: str
    ) -> Tensor:
        path = self._feature_h5_path(t, frame_i, module_path, feature_name)
        return self.read_dataset_np(path)
