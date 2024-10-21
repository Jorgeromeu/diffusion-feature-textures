import h5py
import torch

from text3d2video.artifacts.animation_artifact import ArtifactWrapper


class H5Artifact(ArtifactWrapper):
    wandb_artifact_type = "h5_data"

    def h5_file_path(self):
        return self.folder / "data.h5"

    def open_h5_file(self, mode="w"):
        self.h5_file = h5py.File(self.h5_file_path(), mode)

    def close_h5_file(self):
        self.h5_file.close()

    def create_dataset(
        self, path: str, data: torch.Tensor, dim_names: list[str] = None
    ):
        d = self.h5_file.create_dataset(path, data=data.cpu().numpy())

        if dim_names is not None and len(dim_names) != len(data.shape):
            raise ValueError(
                f"dim_names should have the same length as the number of dimensions of the tensor. Got {len(dim_names)} dim_names and {len(data.shape)} dimensions."
            )

        if dim_names:
            for i, label in enumerate(dim_names):
                d.dims[i].label = label

        return d

    def print_datasets(self):
        def print_path(name, obj):
            if isinstance(obj, h5py.Dataset):
                print(name, obj.shape)

        with h5py.File(self.h5_file_path(), "r") as f:
            f.visititems(print_path)
