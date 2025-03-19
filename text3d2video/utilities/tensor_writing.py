from pathlib import Path

import h5py
from rerun import Tensor


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

    def log(self, path: str, tensor: Tensor, **attributes):
        if path not in self.fp:
            self.fp.create_group(path)

        group = self.fp[path]
        keys = list(group.keys())
        step_idx = len(keys)
        val_name = f"val_{step_idx}"

        dataset = group.create_dataset(val_name, data=tensor.cpu().numpy())

        h5_attribs = {"step": step_idx} | attributes

        for k, v in h5_attribs.items():
            dataset.attrs[k] = v

    def attr_keys(self, path: str):
        group = self.fp[path]
        datasets = list(group.keys())

        keys = set()
        for dataset in datasets:
            attrs = dict(group[dataset].attrs)
            dataset_keys = set(attrs.keys())
            keys |= dataset_keys

    def attr_values(self, path: str, key: str):
        group = self.fp[path]
        datasets = list(group.keys())

        values = []
        for dataset in datasets:
            attrs = dict(group[dataset].attrs)
            if key in attrs:
                values.append(attrs[key])

        return values

    def read_datasets(self, path: str):
        datasets = []
        for key in self.fp[path].keys():
            dataset = self.fp[path][key]
            datasets.append(dataset)

        return datasets
