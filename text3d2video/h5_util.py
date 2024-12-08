import h5py
from torch import Tensor


def print_datasets(h5_path: str, parent_path=""):
    def print_path(name: str, obj):
        is_child = name.startswith(parent_path)

        if not is_child:
            return

        if isinstance(obj, h5py.Dataset):
            print(name, obj.shape)

    with h5py.File(h5_path, "r") as f:
        f.visititems(print_path)


def write_tensor_as_dataset(write_file: h5py.File, path: str, tensor: Tensor):
    data = tensor.cpu().numpy()
    write_file.create_dataset(path, data=data)


def read_tensor_from_dataset(h5_path: str, path: str) -> Tensor:
    with h5py.File(h5_path, "r") as f:
        dset = f[path]
        return Tensor(dset)
