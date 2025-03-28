from abc import ABC
from collections import defaultdict
from pathlib import Path
from typing import Dict

import h5py
import torch
from einops import rearrange
from jaxtyping import Float
from rerun import Tensor

from text3d2video.util import dict_map, ordered_sample
from text3d2video.utilities.h5_util import dset_to_pt


class H5Logger(ABC):
    """
    A utility class for reading and writing named values with keys to an hdf5 file
    """

    # hdf5 file path and file pointer
    path: Path
    fp: h5py.File

    # greenlists for various keys
    key_greenlists: dict[str, list]

    @classmethod
    def create_disabled(cls):
        return cls(None, enabled=False)

    def __init__(self, h5_file_path: Path, enabled=True, attr_greenlists=None):
        self.path = h5_file_path
        self.fp = None
        self.enabled = enabled
        if attr_greenlists is None:
            self.key_greenlists = {}

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

    def write(self, name: str, tensor: Tensor, **keys):
        if not self.enabled:
            return

        if keys is None:
            keys = {}

        # check keys in greenlists
        for key, value in keys.items():
            greenlist = self.key_greenlists.get(key)

            # if no greenlist for key, skip
            if greenlist is None:
                continue

            # if value not in greenlist, dont write
            if value not in greenlist:
                return

        # get filename from keys and name
        filename = self._get_filename(name, keys)

        # create dataset
        data = tensor.cpu().numpy()

        try:
            dataset = self.fp.create_dataset(filename, data=data)
        except ValueError as e:
            keys_str = " ".join(f"{k}={v}" for k, v in keys.items())
            raise ValueError(
                f"Failed to write tensor with name {name} and keys {keys_str}"
            ) from None

        # save all keys as attrs
        for key, value in keys.items():
            dataset.attrs[key] = value

    def read(self, name: str, return_pt=True, check_keys=False, **keys):
        # assert keys contains all field keys
        if check_keys:
            field_keys = self.field_keys(name)
            for field_key in field_keys:
                assert (
                    field_key in keys
                ), f"Failed to specify key {field_key} for field {name}"

        filename = self._get_filename(name, keys)
        dataset = self.fp[filename]

        if return_pt:
            return dset_to_pt(dataset)

        return dataset

    def fields(self):
        return list(self.fp.keys())

    def field_keys(self, name: str):
        group = self.fp[name]

        all_keys = set()

        def visit(path, obj):
            attrs = dict(obj.attrs)
            all_keys.update(attrs.keys())

        group.visititems(visit)
        return list(all_keys)

    def key_values(self, name: str, key: str):
        group = self.fp[name]

        all_vals = set()

        def visit(path, obj):
            val = obj.attrs.get(key)

            if val is not None:
                all_vals.add(val)

        group.visititems(visit)
        return list(all_vals)

    def field_keys_and_vals(self, name: str):
        group = self.fp[name]

        keys_and_vals = defaultdict(lambda: set())

        def visit(path, obj):
            attrs = dict(obj.attrs)

            for k, v in attrs.items():
                keys_and_vals[k].add(v)

        group.visititems(visit)

        result = dict_map(keys_and_vals, lambda _, v: list(v))

        return result


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


class AttnWriter(Writer):
    def write_qkv(
        self,
        qry: Float[Tensor, "b t d"],
        key: Float[Tensor, "b t d"],
        val: Float[Tensor, "b t d"],
        t: int,
        layer: str,
        n_chunks: int = 1,
        frame_indices=None,
        chunk_labels=None,
    ):
        batch_size = qry.shape[0]
        n_frames = batch_size // n_chunks

        if frame_indices is None:
            frame_indices = torch.arange(n_frames)

        if chunk_labels is None:
            if n_chunks == 1:
                chunk_labels = [""]
            else:
                chunk_labels = [f"chunk_{i}" for i in range(n_chunks)]

        assert len(chunk_labels) == n_chunks, "chunk_labels must have length n_chunks"
        assert len(frame_indices) == n_frames, "frame_indices must have length n_frames"
        assert t is not None, "t must be provided"

        qrys_stacked = rearrange(
            qry, "(n_chunks b) t d -> n_chunks b t d", n_chunks=n_chunks
        )
        keys_stacked = rearrange(
            key, "(n_chunks b) t d -> n_chunks b t d", n_chunks=n_chunks
        )
        vals_stacked = rearrange(
            val, "(n_chunks b) t d -> n_chunks b t d", n_chunks=n_chunks
        )

        for chunk_i in range(n_chunks):
            chunk_label = chunk_labels[chunk_i]
            for frame_i in range(n_frames):
                frame_index = frame_indices[frame_i]

                q = qrys_stacked[chunk_i, frame_i]
                self.logger.write(
                    "qry",
                    q,
                    t=t,
                    layer=layer,
                    frame_i=frame_index,
                    chunk=chunk_label,
                )

                k = keys_stacked[chunk_i, frame_i]
                self.logger.write(
                    "key",
                    k,
                    t=t,
                    layer=layer,
                    frame_i=frame_index,
                    chunk=chunk_label,
                )

                v = vals_stacked[chunk_i, frame_i]
                self.logger.write(
                    "val",
                    v,
                    t=t,
                    layer=layer,
                    frame_i=frame_index,
                    chunk=chunk_label,
                )


class GrLogger(H5Logger):
    def __init__(
        self, h5_file_path: Path, enabled=True, n_save_times=10, n_save_frames=10
    ):
        super().__init__(h5_file_path, enabled)
        self.n_save_times = n_save_times
        self.n_save_frames = n_save_frames
        self.latents_writer = LatentsWriter(self)
        self.attn_writer = AttnWriter(self)

    def setup_greenlists(self, denoising_times: list[int], n_frames: int):
        # setup time greenlist
        if self.n_save_times is not None:
            all_noise_levels = denoising_times + [0]
            self.key_greenlists["t"] = ordered_sample(
                all_noise_levels, self.n_save_times
            )

        # setup frame greenlist
        if self.n_save_frames is not None:
            self.key_greenlists["frame_i"] = ordered_sample(
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


class FeatureExtractionLogger(H5Logger):
    def write_features_dict(self, name: str, features: Dict[str, Tensor], **keys):
        for layer, x in features.items():
            self.write(name, x, layer=layer, **keys)
