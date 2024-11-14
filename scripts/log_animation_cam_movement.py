import tempfile
from pathlib import Path

import click
import torch
from pytorch3d.renderer import FoVPerspectiveCameras

import wandb
from text3d2video.artifacts.animation_artifact import AnimationArtifact
from text3d2video.artifacts.video_artifact import pil_frames_to_clip
from text3d2video.camera_placement import (
    sideways_orthographic_cameras,
    turntable_cameras,
    z_movement_cameras,
)
from text3d2video.rendering import render_depth_map


def log_animation(
    artifact_name: str, static_path: Path, cameras: FoVPerspectiveCameras
):
    wandb.init(project="diffusion-3D-features", job_type="log_artifact")
    artifact: AnimationArtifact = AnimationArtifact.create_empty_artifact(artifact_name)
    artifact.write_cameras(cameras)
    artifact.write_unposed(static_path)

    clip = artifact.render_depth_clip()

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=True) as f:
        temp_filename = f.name
        print(temp_filename)
        clip.write_videofile(temp_filename, codec="libx264", fps=10)
        wandb.log({"animation": wandb.Video(temp_filename)})

    artifact.log_if_enabled()
    wandb.finish()


@click.group()
@click.option("--artifact_name", type=str, required=True)
@click.option("--mesh_path", type=click.Path(exists=True), required=True)
@click.option("--n_frames", type=int, required=False, default=10)
@click.pass_context
def cli(ctx, artifact_name, mesh_path, n_frames):
    ctx.obj["artifact_name"] = artifact_name
    ctx.obj["mesh_path"] = mesh_path
    ctx.obj["n_frames"] = n_frames


@click.command()
@click.option("--dist", type=float, default=2)
@click.option("--start_angle", type=float, default=0)
@click.option("--stop_angle", type=float, default=360)
@click.pass_context
def turntable(ctx, dist, start_angle, stop_angle):
    artifact_name = ctx.obj["artifact_name"]
    mesh_path = ctx.obj["mesh_path"]
    n_frames = ctx.obj["n_frames"]
    cams = turntable_cameras(
        n=n_frames, dist=dist, start_angle=start_angle, stop_angle=stop_angle
    )
    log_animation(artifact_name, mesh_path, cams)


@click.command()
@click.option("--x_start", type=float, default=1)
@click.option("--x_stop", type=float, default=-1)
@click.pass_context
def sideways_orth(ctx, x_start, x_stop):
    artifact_name = ctx.obj["artifact_name"]
    mesh_path = ctx.obj["mesh_path"]
    n_frames = ctx.obj["n_frames"]
    cams = sideways_orthographic_cameras(n=n_frames, x_0=x_start, x_1=x_stop)
    log_animation(artifact_name, mesh_path, cams)


@click.command()
@click.option("--z_start", type=float, default=2)
@click.option("--z_stop", type=float, default=4)
@click.pass_context
def z_movement(ctx, x_start, x_stop):
    artifact_name = ctx.obj["artifact_name"]
    mesh_path = ctx.obj["mesh_path"]
    n_frames = ctx.obj["n_frames"]
    cams = z_movement_cameras(n=n_frames, z_0=x_start, z_1=x_stop)
    log_animation(artifact_name, mesh_path, cams)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    cli.add_command(turntable)
    cli.add_command(sideways_orth)
    cli.add_command(z_movement)
    cli(obj={})
