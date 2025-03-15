from pathlib import Path

import h5py
import torch
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from torch import Tensor

from text3d2video.util import assert_tensor_shape, assert_tensor_shapes, ordered_sample
from text3d2video.utilities.h5_util import (
    print_datasets,
    read_tensor_from_dataset,
    write_tensor_as_dataset,
)


class DiffusionDataLogger:
    """
    Class to manage reading and writing data extracted during diffusion inference to disk
    """

    # save config
    enabled: bool
    path_greenlist: list[str]
    noise_level_greenlist: list[int]
    frame_indices_greenlist: list[int]

    # h5 file path and file pointer
    h5_file_path: Path
    h5_write_fp: h5py.File

    def __init__(
        self,
        h5_file_path: Path,
        enabled=True,
        path_greenlist: list[str] = None,
        frame_indices_greenlist: list[int] = None,
        noise_level_greenlist: list[int] = None,
    ):
        # default to empty list
        if path_greenlist is None:
            path_greenlist = []
        if frame_indices_greenlist is None:
            frame_indices_greenlist = []
        if noise_level_greenlist is None:
            noise_level_greenlist = []

        self.h5_file_path = h5_file_path

        # save config
        self.enabled = enabled
        self.path_greenlist = path_greenlist
        self.frame_indices_greenlist = frame_indices_greenlist
        self.noise_level_greenlist = noise_level_greenlist

        # initialize to None
        self.h5_write_fp = None

    def calc_evenly_spaced_noise_noise_levels(
        self, scheduler: SchedulerMixin, n_levels: int = -1
    ):
        """
        Calculate the noise levels to save based on the scheduler and the number of levels to save
        """

        all_noise_levels = [int(t) for t in scheduler.timesteps] + [0]

        # use all timesteps
        if n_levels == -1:
            self.noise_level_greenlist = all_noise_levels
            return

        # use ordered sample
        self.noise_level_greenlist = ordered_sample(all_noise_levels, n_levels)

    def calc_evenly_spaced_frame_indices(self, n_frames: int, n_save_frames: int = -1):
        """
        Calculate the frames to save based on the number of frames and the number of frames to save
        """

        all_frame_indices = list(range(n_frames))

        if n_save_frames == -1:
            self.frame_indices_greenlist = all_frame_indices
            return

        self.frame_indices_greenlist = ordered_sample(all_frame_indices, n_save_frames)

    def _in_greenlist(self, x, greenlist):
        return greenlist == [] or x in greenlist

    def should_write(self, t: int = None, frame_i: int = None, attn_path: str = None):
        # only write if for each provided index, it is in greenlist, or greenlist is empty

        valid_t = t is None or self._in_greenlist(t, self.noise_level_greenlist)
        valid_frame = frame_i is None or self._in_greenlist(
            frame_i, self.frame_indices_greenlist
        )
        valid_path = attn_path is None or self._in_greenlist(
            attn_path, self.path_greenlist
        )

        return valid_t and valid_frame and valid_path

    def print_datasets(self, prefix: str = ""):
        print_datasets(self.h5_file_path, parent_path=prefix)

    def memory_usage(self):
        gbs = self.h5_file_path.stat().st_size
        return gbs / 1e9

    def begin_recording(self):
        self.h5_write_fp = h5py.File(self.h5_file_path, "w")

    def end_recording(self):
        if self.h5_write_fp is None:
            return
        self.h5_write_fp.close()

    def delete_data(self):
        if self.h5_file_path.exists():
            self.h5_file_path.unlink()


# Implement various data writers here


class DiffusionDataWriter:
    """
    Abstract base class for a diffusion data writer
    """

    diff_data: DiffusionDataLogger

    def __init__(self, diff_data: DiffusionDataLogger, data_path: str):
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
        if self.diff_data.should_write(
            t=timestep, frame_i=frame_i, attn_path=attn_path
        ):
            write_tensor_as_dataset(self.diff_data.h5_write_fp, path, data)

    def _read_tensor(self, path: str):
        return read_tensor_from_dataset(self.diff_data.h5_file_path, path)


class LatentsWriter(DiffusionDataWriter):
    """
    Class to write latents to diffusion data
    """

    enabled: bool

    def __init__(
        self, diff_data: DiffusionDataLogger, enabled=True, data_path: str = "latents"
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
        for i in self.diff_data.frame_indices_greenlist:
            self.write_latent(t, i, latents[i])

    def read_latent(self, t: int, frame_i: int):
        path = self._latent_path(t, frame_i)
        return self._read_tensor(path)

    def read_latents_at_frame(self, frame_i: int):
        times = self.diff_data.save_times
        return torch.stack([self.read_latent(t, frame_i) for t in times])

    def read_latents_at_time(self, t: int):
        frame_indices = self.diff_data.frame_indices_greenlist
        return torch.stack([self.read_latent(t, frame_i) for frame_i in frame_indices])


class AttnFeaturesWriter(DiffusionDataWriter):
    """
    Class to write attention data
    """

    def __init__(
        self,
        diff_data,
        save_q=True,
        save_k=True,
        save_v=True,
        save_y=True,
        data_path="attn_data",
    ):
        super().__init__(diff_data, data_path)
        self.save_q = save_q
        self.save_k = save_k
        self.save_v = save_v
        self.save_y = save_y

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

    def write_attn_out_batched(
        self, t: int, layer: str, y: Tensor, chunk_frame_indices: Tensor
    ):
        if chunk_frame_indices is None:
            chunk_frame_indices = torch.arange(y.shape[0])
        assert_tensor_shapes([(y, ("B", "F", "T_kv", "D_v"))])

        if not self.save_y:
            return

        for tensor_idx, frame_i in enumerate(chunk_frame_indices):
            self.write_seq(t, frame_i, layer, "attn_out", y[0, tensor_idx, ...])

    # reading

    def read_attn_out(self, t: int, frame_i: int, layer: str):
        path = self._seq_path(t, frame_i, layer, "attn_out")
        return self._read_tensor(path)

    def read_qry(self, t: int, frame_i: int, layer: str):
        path = self._seq_path(t, frame_i, layer, "qry")
        return self._read_tensor(path)

    def read_key(self, t: int, frame_i: int, layer: str):
        path = self._seq_path(t, frame_i, layer, "key")
        return self._read_tensor(path)

    def read_val(self, t: int, frame_i: int, layer: str):
        path = self._seq_path(t, frame_i, layer, "val")
        return self._read_tensor(path)
