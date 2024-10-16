import h5py
import torch

from text3d2video.artifacts.animation_artifact import ArtifactWrapper


class TensorsArtifact(ArtifactWrapper):

    wandb_artifact_type = "tensors"

    def h5_file_path(self):
        return self.folder / "data.h5"

    def open_h5_file(self, mode="w"):
        self.h5_file = h5py.File(self.h5_file_path(), mode)

    def close_h5_file(self):
        self.h5_file.close()

    def save_tensor(self, name: str, tensor: torch.Tensor):
        self.h5_file.create_dataset(name, data=tensor.cpu().numpy())
