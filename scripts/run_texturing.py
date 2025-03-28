import torch
from attr import dataclass

import wandb_util.wandb_util as wbu
from text3d2video.artifacts.mesh_artifact import MeshArtifact
from text3d2video.pipelines.pipeline_utils import ModelConfig
from text3d2video.pipelines.texturing_pipeline import TexturingConfig


@dataclass
class RunTexturing:
    run: wbu.RunConfig
    prompt: str
    mesh_art: str
    texturing_config: TexturingConfig
    model: ModelConfig
    seed: int


class RunTexturing(wbu.WandbRun):
    job_type = "run_texturing"

    def _run(self, cfg: RunTexturing):
        torch.set_grad_enabled(False)

        # read mesh
        mesh = MeshArtifact.from_wandb_artifact_tag(
            cfg.mesh_art, download=cfg.run.download_artifacts
        )
