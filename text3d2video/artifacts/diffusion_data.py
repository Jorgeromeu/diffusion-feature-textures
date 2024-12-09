from dataclasses import dataclass
from pathlib import Path

import h5py
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from torch import Tensor

from text3d2video.h5_util import read_tensor_from_dataset, write_tensor_as_dataset
from text3d2video.util import assert_tensor_shape, ordered_sample
from text3d2video.wandb_util import ArtifactWrapper


@dataclass
class DiffusionDataCfg:
    enabled: bool
    n_save_steps: int
    n_save_frames: int
    attn_paths: list[str]


class DiffusionData:
    """
    Class to manage saving diffusion data
    """

    config: DiffusionDataCfg
    _save_step_times: list[int]
    _save_frame_indices: list[int]
    h5_file_path: Path
    h5_write_fp: h5py.File

    def __init__(self, config: DiffusionDataCfg, h5_file_path: Path):
        self.config = config
        self.h5_file_path = h5_file_path

        # intiialize to empty
        self._save_frame_indices = []
        self._save_step_times = []

        # initialize to None
        self.h5_write_fp = None

    def calculate_save_steps(self, scheduler: SchedulerMixin):
        n_save_steps = self.config.n_save_steps
        all_timesteps = [int(t) for t in scheduler.timesteps]

        if n_save_steps == -1:
            self._save_step_times = all_timesteps
            return

        self._save_step_times = ordered_sample(
            all_timesteps[:-1], self.config.n_save_steps
        )

    def calculate_save_frames(self, n_frames: int):
        n_save_frames = self.config.n_save_frames
        all_frame_indices = list(range(n_frames))

        if n_save_frames == -1:
            self._save_frame_indices = all_frame_indices
            return

        self._save_frame_indices = ordered_sample(all_frame_indices, n_save_frames)

    def should_save(self, t: int = None, frame_i: int = None, attn_path: str = None):
        valid_timestep = t is None or t in self.save_times
        valid_frame = frame_i is None or frame_i in self.save_frame_indices
        valid_attn_path = attn_path is None or attn_path in self.config.attn_paths

        return (
            self.config.enabled and valid_timestep and valid_frame and valid_attn_path
        )

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
        return self.config.attn_paths

    def begin_recording(self):
        self.h5_write_fp = h5py.File(self.h5_file_path, "w")

    def end_recording(self):
        if self.h5_write_fp is None:
            return
        self.h5_write_fp.close()


class DiffusionDataWriter:
    diff_data: DiffusionData

    def __init__(self, diff_data: DiffusionData, data_path: str):
        self.diff_data = diff_data
        self.data_path = data_path


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
        if self.enabled and self.diff_data.should_save(t=t, frame_i=frame_i):
            path = self._latent_path(t, frame_i)
            write_tensor_as_dataset(self.diff_data.h5_write_fp, path, latent)

    def write_latents_batched(self, t: int, latents: Tensor):
        assert_tensor_shape(latents, ("B", "C", "H", "W"))
        for i in self.diff_data.save_frame_indices:
            self.write_latent(t, i, latents[i])

    def read_latent(self, t: int, frame_i: int):
        path = self._latent_path(t, frame_i)
        return read_tensor_from_dataset(self.diff_data.h5_file_path, path)


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

        if not self.diff_data.should_save(t=t, frame_i=frame_i, attn_path=layer):
            return

        path = self._seq_path(t, frame_i, layer, seq_name)
        write_tensor_as_dataset(self.diff_data.h5_write_fp, path, seq)

    def write_qkv_batched(self, t: int, layer: str, q: Tensor, k: Tensor, v: Tensor):
        assert_tensor_shape(q, ("B", "T", "D"))
        assert_tensor_shape(k, ("B", "T", "D"))
        assert_tensor_shape(v, ("B", "T", "D"))

        for frame_i in self.diff_data.save_frame_indices:
            if self.save_q:
                self.write_seq(t, frame_i, layer, "qry", q[frame_i])
            if self.save_k:
                self.write_seq(t, frame_i, layer, "key", k[frame_i])
            if self.save_v:
                self.write_seq(t, frame_i, layer, "val", v[frame_i])

    # reading

    def read_qry(self, t: int, frame_i: int, layer: str):
        path = self._seq_path(t, frame_i, layer, "qry")
        return read_tensor_from_dataset(self.diff_data.h5_file_path, path)

    def read_key(self, t: int, frame_i: int, layer: str):
        path = self._seq_path(t, frame_i, layer, "key")
        return read_tensor_from_dataset(self.diff_data.h5_file_path, path)

    def read_val(self, t: int, frame_i: int, layer: str):
        path = self._seq_path(t, frame_i, layer, "val")
        return read_tensor_from_dataset(self.diff_data.h5_file_path, path)
