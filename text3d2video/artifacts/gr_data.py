import h5py
import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from torch import Tensor

from text3d2video.artifacts.tensors_artifact import H5Artifact
from text3d2video.generative_rendering.configs import GrSaveConfig
from text3d2video.util import ordered_sample, ordered_sample_indices


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

    # Writing methods

    def setup_save_config_write(
        self, cfg: GrSaveConfig, scheduler: SchedulerMixin, n_frames: int
    ):
        self.config = cfg

        if not self.config.enabled:
            self.save_frame_indices = []
            self.save_timesteps = []
            return

        # compute frames to save
        frame_indices = list(range(n_frames))
        self.save_frame_indices = ordered_sample_indices(
            frame_indices, self.config.n_frames
        )

        # compute diffusion steps to save
        diffusion_steps = torch.cat([scheduler.timesteps, torch.Tensor([0])])
        self.save_timesteps = ordered_sample(diffusion_steps, self.config.n_timesteps)

        # open h5 file
        self.open_h5_file()

        # save the saved frame and tinstep indices
        frame_indices_tensor = Tensor(self.save_frame_indices)
        timesteps_tensor = Tensor(self.save_timesteps)

        self.create_dataset("frame_indices", frame_indices_tensor)
        self.create_dataset("timesteps", timesteps_tensor)

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
        indices = self.read_dataset("frame_indices").long()
        return indices.tolist()

    def read_timesteps(self):
        timesteps = self.read_dataset("timesteps").long()
        return timesteps.tolist()

    def read_module_paths(self):
        module_paths = set()

        def find_layer_name(name, obj):
            if isinstance(obj, h5py.Dataset):
                path = name.split("/")
                path_len = len(path)
                if path_len == 4:
                    path_layer = path[2]
                    layer = path_layer.split("_", 1)[1]
                    module_paths.add(layer)

        with h5py.File(self.h5_file_path(), "r") as f:
            f.visititems(find_layer_name)

        return sorted(list(module_paths))

    def read_latent(self, t: int, frame_i: int) -> Tensor:
        path = self._latent_h5_path(t, frame_i)
        return self.read_dataset(path)

    def read_feature(
        self, t: int, frame_i: int, module_path: str, feature_name: str
    ) -> Tensor:
        path = self._feature_h5_path(t, frame_i, module_path, feature_name)
        return self.read_dataset(path)
