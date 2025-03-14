from text3d2video.artifacts.anim_artifact import AnimationArtifact


def test_something():
    anim = AnimationArtifact.from_wandb_artifact_tag("rumba:latest")
    print(anim)
