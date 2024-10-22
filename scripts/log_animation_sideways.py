from pathlib import Path

import click
import torch
from pytorch3d.renderer import FoVOrthographicCameras, FoVPerspectiveCameras

import wandb
from text3d2video.artifacts.animation_artifact import AnimationArtifact


def log_animation(
    artifact_name: str, static_path: Path, cameras: FoVPerspectiveCameras
):
    wandb.init(project="diffusion-3D-features", job_type="log_artifact")
    artifact: AnimationArtifact = AnimationArtifact.create_empty_artifact(artifact_name)
    artifact.write_cameras(cameras)
    artifact.write_unposed(static_path)
    artifact.log_if_enabled()
    wandb.finish()


@click.command()
@click.option("--artifact_name", type=str, required=True)
@click.option("--mesh_path", type=click.Path(exists=True), required=True)
@click.option("--n_frames", type=int, required=False, default=100)
def log_animation_main(artifact_name: str, mesh_path: Path, n_frames: int):
    x0 = 1
    x1 = -1

    line = torch.linspace(x0, x1, n_frames)

    r = torch.eye(3)
    r[0, 0] = -1
    r[2, 2] = -1

    R = r.repeat(n_frames, 1, 1)

    T = torch.zeros(n_frames, 3)
    T[:, 2] = 2
    T[:, 0] = line

    device = torch.device("cuda")
    cameras = FoVOrthographicCameras(device=device, R=R, T=T)

    log_animation(artifact_name, mesh_path, cameras)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    log_animation_main()
