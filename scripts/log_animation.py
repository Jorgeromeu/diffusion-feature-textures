from pathlib import Path
import wandb

from text3d2video.wandb_util import AnimationArtifact


def log_animation(animation_name, animation_path, static_path):
    wandb.init(project="diffusion-3D-features", job_type='log artifact')
    artifact = AnimationArtifact.create(
        animation_name,
        animation_path,
        static_path
    )
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    animation_name = 'dancing'
    animation_path = 'data/dancing/'
    static_path = 'data/mixamo-human.obj'
    log_animation(animation_name, animation_path, static_path)
