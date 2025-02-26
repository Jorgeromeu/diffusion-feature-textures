from pathlib import Path

import click

import wandb
from text3d2video.artifacts.animation_artifact import AnimationArtifact


def log_animation(artifact_name: str, animation_path: Path, static_path: Path):
    wandb.init(project="diffusion-3D-features", job_type="log_artifact")
    artifact: AnimationArtifact = AnimationArtifact.create_empty_artifact(artifact_name)
    artifact.write_unposed(static_path)
    artifact.write_frames(animation_path)
    artifact.log_if_enabled()
    wandb.finish()


@click.command()
@click.option("--artifact_name", type=str, required=True)
@click.option("--animation_path", type=click.Path(exists=True), required=True)
@click.option("--static_path", type=click.Path(exists=True), required=True)
def log_animation_main(artifact_name: str, animation_path: Path, static_path: Path):
    """
    Log an animation artifact to wandb
    """
    log_animation(artifact_name, animation_path, static_path)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    log_animation_main()
