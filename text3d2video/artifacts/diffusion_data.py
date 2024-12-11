from dataclasses import dataclass
from pathlib import Path

import h5py
import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from networkx import numeric_assortativity_coefficient
from torch import Tensor

from text3d2video.h5_util import (
    print_datasets,
    read_tensor_from_dataset,
    write_tensor_as_dataset,
)
from text3d2video.util import assert_tensor_shape, assert_tensor_shapes, ordered_sample


@dataclass
class DiffusionDataCfg:
    enabled: bool
    n_save_steps: int
    n_save_frames: int
    attn_paths: list[str]


class DiffusionData:
    """
    Class to manage reading and writing data extracted during diffusion inference
    """

    enabled: bool
    n_save_steps: int
    n_save_frames: int
    attn_paths: list[str]

    _save_step_times: list[int]
    _save_frame_indices: list[int]
    h5_file_path: Path
    h5_write_fp: h5py.File

    def __init__(
        self,
        h5_file_path: Path,
        enabled=True,
        n_save_steps: int = -1,
        n_save_frames: int = -1,
        attn_paths: list[str] = [],
    ):
        self.h5_file_path = h5_file_path

        # save config
        self.enabled = enabled
        self.n_save_steps = n_save_steps
        self.n_save_frames = n_save_frames
        self.attn_paths = attn_paths

        # intiialize to empty
        self._save_frame_indices = []
        self._save_step_times = []

        # initialize to None
        self.h5_write_fp = None

    def calculate_save_steps(self, scheduler: SchedulerMixin):
        n_save_steps = self.n_save_steps
        all_timesteps = [int(t) for t in scheduler.timesteps]

        if n_save_steps == -1:
            self._save_step_times = all_timesteps
            return

        self._save_step_times = ordered_sample(all_timesteps[:-1], self.n_save_steps)

    def calculate_save_frames(self, n_frames: int):
        n_save_frames = self.n_save_frames
        all_frame_indices = list(range(n_frames))

        if n_save_frames == -1:
            self._save_frame_indices = all_frame_indices
            return

        self._save_frame_indices = ordered_sample(all_frame_indices, n_save_frames)

    def should_save(self, t: int = None, frame_i: int = None, attn_path: str = None):
        valid_timestep = t is None or t in self.save_times
        valid_frame = frame_i is None or frame_i in self.save_frame_indices
        valid_attn_path = attn_path is None or attn_path in self.attn_paths

        return self.enabled and valid_timestep and valid_frame and valid_attn_path

    def print_datasets(self, prefix: str = ""):
        print_datasets(self.h5_file_path, parent_path=prefix)

    @property
    def save_times(self):
        """
        Return the diffusion times at which we save data
        """
        return self._save_step_times + [0]

    @property
    def save_step_times(self):
        """
        Return the timesteps at which we save data
        """
        return self._save_step_times

    @property
    def save_frame_indices(self):
        """
        Return the frame indices at which we save data
        """
        return self._save_frame_indices

    @property
    def save_module_paths(self):
        """
        Return the module paths at which we save data
        """
        return self.attn_paths

    def begin_recording(self):
        self.h5_write_fp = h5py.File(self.h5_file_path, "w")

    def end_recording(self):
        if self.h5_write_fp is None:
            return
        self.h5_write_fp.close()


class DiffusionDataWriter:
    """
    Abstract base class for a diffusion data writer
    """

    diff_data: DiffusionData

    def __init__(self, diff_data: DiffusionData, data_path: str):
        self.diff_data = diff_data
        self.data_path = data_path

    def write_tensor(
        self,
        path: str,
        data: Tensor,
        timestep: int = None,
        frame_i: int = None,
        attn_path: str = None,
    ):
        if self.diff_data.should_save(t=timestep, frame_i=frame_i, attn_path=attn_path):
            write_tensor_as_dataset(self.diff_data.h5_write_fp, path, data)

    def read_tensor(self, path: str):
        return read_tensor_from_dataset(self.diff_data.h5_file_path, path)


class LatentsWriter(DiffusionDataWriter):
    """
    Class to write latents to diffusion data
    """

    enabled: bool

    def __init__(
        self, diff_data: DiffusionData, enabled=True, data_path: str = "latents"
    ):
        super().__init__(diff_data, data_path)
        self.enabled = enabled

    def _latent_path(self, t: int, frame_i: int):
        return f"{self.data_path}/time_{t}/frame_{frame_i}"

    def write_latent(self, t: int, frame_i: int, latent: Tensor):
        assert_tensor_shape(latent, ("C", "H", "W"))
        if self.enabled:
            path = self._latent_path(t, frame_i)
            self.write_tensor(path, latent, timestep=t, frame_i=frame_i)

    def write_latents_batched(self, t: int, latents: Tensor):
        assert_tensor_shape(latents, ("B", "C", "H", "W"))
        for i in self.diff_data.save_frame_indices:
            self.write_latent(t, i, latents[i])

    def read_latent(self, t: int, frame_i: int):
        path = self._latent_path(t, frame_i)
        return self.read_tensor(path)

    def read_latents_at_frame(self, frame_i: int):
        times = self.diff_data.save_times
        return torch.stack([self.read_latent(t, frame_i) for t in times])

    def read_latents_at_time(self, t: int):
        frame_indices = self.diff_data.save_frame_indices
        return torch.stack([self.read_latent(t, frame_i) for frame_i in frame_indices])


class AttnFeaturesWriter(DiffusionDataWriter):
    """
    Class to write attention data
    """

    def __init__(
        self, diff_data, save_q=True, save_k=True, save_v=True, data_path="attn_data"
    ):
        super().__init__(diff_data, data_path)
        self.save_q = save_q
        self.save_k = save_k
        self.save_v = save_v

    def _seq_path(self, t: int, frame_i: int, layer: str, seq_name: str):
        return f"{self.data_path}/time_{t}/frame_{frame_i}/{layer}/{seq_name}"

    # writing

    def write_seq(self, t: int, frame_i: int, layer: str, seq_name: str, seq: Tensor):
        assert_tensor_shape(seq, ("T", "D"))
        path = self._seq_path(t, frame_i, layer, seq_name)
        self.write_tensor(path, seq, frame_i=frame_i, attn_path=layer, timestep=t)

    def write_qkv_batched(
        self,
        t: int,
        layer: str,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        chunk_frame_indices: Tensor = None,
    ):
        if chunk_frame_indices is None:
            chunk_frame_indices = torch.arange(q.shape[0])

        assert_tensor_shapes(
            [
                (q, ("B", "F", "T_q", "D_qk")),
                (k, ("B", "F", "T_kv", "D_qk")),
                (v, ("B", "F", "T_kv", "D_v")),
                (chunk_frame_indices, ("F",)),
            ]
        )

        for tensor_idx, frame_i in enumerate(chunk_frame_indices):
            if self.save_q:
                self.write_seq(t, frame_i, layer, "qry", q[0, tensor_idx, ...])
            if self.save_k:
                self.write_seq(t, frame_i, layer, "key", k[0, tensor_idx, ...])
            if self.save_v:
                self.write_seq(t, frame_i, layer, "val", v[0, tensor_idx, ...])

    # reading

    def read_qry(self, t: int, frame_i: int, layer: str):
        path = self._seq_path(t, frame_i, layer, "qry")
        return self.read_tensor(path)

    def read_key(self, t: int, frame_i: int, layer: str):
        path = self._seq_path(t, frame_i, layer, "key")
        return self.read_tensor(path)

    def read_val(self, t: int, frame_i: int, layer: str):
        path = self._seq_path(t, frame_i, layer, "val")
        return self.read_tensor(path)
