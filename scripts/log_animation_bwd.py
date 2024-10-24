from pathlib import Path

import click
import torch
from pytorch3d.renderer import FoVPerspectiveCameras

import wandb
from text3d2video.artifacts.animation_artifact import AnimationArtifact
from text3d2video.camera_placement import z_movement_cameras


def log_animation(artifact_name: str, static_path: Path, cameras: FoVPerspectiveCameras):
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
    device = torch.device("cuda")
    cameras = z_movement_cameras(z_0=2, z_1=4, n=n_frames, device=device)
    log_animation(artifact_name, mesh_path, cameras)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    log_animation_main()
