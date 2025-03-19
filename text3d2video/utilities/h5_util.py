import h5py
import torch
from torch import Tensor


def print_datasets(h5_path: str, parent_path="", leaves_only=True):
    def print_path(path: str, obj):
        is_leaf = path.startswith(parent_path)

        if not is_leaf:
            if leaves_only:
                return
            else:
                print(path)

        if isinstance(obj, h5py.Dataset):
            print(path, obj.shape, dict(obj.attrs))

    with h5py.File(h5_path, "r") as f:
        f.visititems(print_path)


def write_tensor_as_dataset(write_file: h5py.File, path: str, tensor: Tensor):
    data = tensor.cpu().numpy()
    write_file.create_dataset(path, data=data)


def read_tensor_from_dataset(h5_path: str, path: str) -> Tensor:
    with h5py.File(h5_path, "r") as f:
        dset = f[path]
        return Tensor(dset)


def dataset_to_tensor(dataset) -> Tensor:
    return torch.from_numpy(dataset[:])
