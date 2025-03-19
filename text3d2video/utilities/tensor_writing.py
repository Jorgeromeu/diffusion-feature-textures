from pathlib import Path

import h5py
from torch import Tensor

from text3d2video.utilities.h5_util import dset_to_pt


class H5logger:
    path: Path
    fp: h5py.File

    def __init__(self, h5_file_path: Path):
        self.path = h5_file_path
        self.fp = None

    def open_write(self):
        self.fp = h5py.File(self.path, "w")

    def open_read(self):
        self.fp = h5py.File(self.path, "r")

    def close(self):
        if self.fp is not None:
            self.fp.close()

    def delete_data(self):
        if self.path.exists():
            self.path.unlink()

    def write_dataset(
        self,
        path: str,
        tensor: Tensor,
        attrs: dict = None,
    ):
        tensor_np = tensor.cpu().numpy()
        dataset = self.fp.create_dataset(path, data=tensor_np)

        if attrs is not None:
            for key, value in attrs.items():
                dataset.attrs[key] = value

    def read_dataset(self, path: str) -> h5py.Dataset:
        return self.fp[path]


class FeatureLogger(H5logger):
    def _path(self, layer: str, t: int, name: str):
        return f"features/{layer}/t_{t}/{name}"

    def write(self, layer: str, t: int, name: str, tensor: Tensor):
        attrs = {"t": t, "layer": layer}
        self.write_dataset(self._path(layer, t, name), tensor, attrs)

    def read(self, layer: str, t: int, name: str) -> Tensor:
        dset = self.read_dataset(self._path(layer, t, name))
        return dset_to_pt(dset)
