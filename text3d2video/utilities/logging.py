from abc import ABC
from pathlib import Path
from typing import Dict

import h5py
from rerun import Tensor

from text3d2video.util import ordered_sample
from text3d2video.utilities.h5_util import dset_to_pt


class H5Logger(ABC):
    """
    A utility class for reading/writing tensors over various time indices to hdf5
    """

    path: Path
    fp: h5py.File
    keys_greenlists: dict[str, list]

    @classmethod
    def create_disabled(cls):
        return cls(None, enabled=False)

    def __init__(self, h5_file_path: Path, enabled=True, attr_greenlists=None):
        self.path = h5_file_path
        self.fp = None
        self.enabled = enabled
        if attr_greenlists is None:
            self.keys_greenlists = {}

    def open_write(self):
        if not self.enabled:
            return

        self.fp = h5py.File(self.path, "w")

    def open_read(self):
        if not self.enabled:
            return

        self.fp = h5py.File(self.path, "r")

    def close(self):
        if not self.enabled:
            return

        if self.fp is not None:
            self.fp.close()

    def delete_data(self):
        if not self.enabled:
            return

        if self.path.exists():
            self.path.unlink()

    def _get_filename(self, name, attrs: Dict):
        if attrs == {}:
            identifier = "value"

        else:
            attrs_sorted = sorted(attrs.items(), key=lambda x: x[0])
            identifier = "value_" + "&".join(f"{k}_{v}" for k, v in attrs_sorted)

        return f"{name}/{identifier}"

    def write(self, name: str, tensor: Tensor, attrs: Dict = None, **keys):
        if not self.enabled:
            return

        if keys is None:
            keys = {}
        if attrs is None:
            attrs = {}

        # check keys in greenlists
        for key, value in keys.items():
            greenlist = self.keys_greenlists.get(key)
            if greenlist is None:
                continue

            if value not in greenlist:
                return

        # get filename from keys and name
        filename = self._get_filename(name, keys)

        # create dataset
        data = tensor.cpu().numpy()
        dataset = self.fp.create_dataset(filename, data=data)

        # save all keys as attrs
        for key, value in keys.items():
            dataset.attrs[key] = value

        # save all attrs additionally
        for key, value in attrs.items():
            dataset.attrs[key] = value

    def read(self, name: str, return_pt=False, **keys):
        filename = self._get_filename(name, keys)
        dataset = self.fp[filename]

        if return_pt:
            return dset_to_pt(dataset)

        return dataset

    def value_names(self):
        return list(self.fp.keys())

    def get_keys(self, name: str):
        group = self.fp[name]

        all_keys = set()

        def visit(path, obj):
            attrs = dict(obj.attrs)
            all_keys.update(attrs.keys())

        group.visititems(visit)
        return list(all_keys)

    def get_key_values(self, name: str, key: str):
        group = self.fp[name]

        all_vals = set()

        def visit(path, obj):
            val = obj.attrs.get(key)

            if val is not None:
                all_vals.add(val)

        group.visititems(visit)
        return list(all_vals)


class Writer:
    logger: H5Logger

    def __init__(self, logger: H5Logger):
        self.logger = logger


# Implementations


class LatentsWriter(Writer):
    def write_latents(self, latents: Tensor, t: int):
        for i, latent in enumerate(latents):
            self.logger.write("latent", latent, t=t, frame_i=i)

    def read_latent(self, t: int, frame_i: int):
        return self.logger.read("latent", t=t, frame_i=frame_i)


class GrLogger(H5Logger):
    def __init__(
        self, h5_file_path: Path, enabled=True, n_save_times=10, n_save_frames=10
    ):
        super().__init__(h5_file_path, enabled)
        self.n_save_times = n_save_times
        self.n_save_frames = n_save_frames
        self.latents_writer = LatentsWriter(self)

    def setup_greenlists(self, denoising_times: list[int], n_frames: int):
        # setup time greenlist
        all_noise_levels = denoising_times + [0]
        self.keys_greenlists["t"] = ordered_sample(all_noise_levels, self.n_save_times)

        # setup frame greenlist
        self.keys_greenlists["frame_i"] = ordered_sample(
            range(n_frames), self.n_save_frames
        )

    def write_feature_textures(
        self,
        name: str,
        textures: Dict[str, Tensor],
        t,
    ):
        keys_base = {"t": t}

        for layer, texture in textures.items():
            keys = keys_base | {"layer": layer}
            self.write(name, texture, **keys)

    def write_rendered_features(
        self, name: str, features: Dict[str, Tensor], t, frame_indices: list[int]
    ):
        keys_base = {"t": t}

        for layer, feature_maps in features.items():
            for i, feature_map in enumerate(feature_maps):
                frame_i = frame_indices[i]
                keys = keys_base | {"layer": layer, "frame_i": frame_i}
                self.write(name, feature_map, **keys)
