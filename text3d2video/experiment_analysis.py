from typing import List

from attr import dataclass
from omegaconf import OmegaConf
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.structures import Meshes
from rerun import Image, Tensor

import wandb_util.wandb_util as wbu
from scripts.wandb_runs.run_generative_rendering import RunGenerativeRenderingConfig
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.rendering import render_depth_map, render_rgb_uv_map
from wandb.apis.public import Run


@dataclass
class GrData:
    """
    Dataclass to hold all of the data relevant for analyzing a Generative Rendering run.
    """

    frames: List[Image]
    prompt: str
    gr_config: GenerativeRenderingConfig
    cams: CamerasBase
    meshes: Meshes
    verts_uvs: Tensor
    faces_uvs: Tensor
    uvs: List[Image] = None
    depths: List[Image] = None

    @classmethod
    def from_gr_run(
        cls, run: Run, with_video: bool = True, with_anim: bool = True
    ) -> "GrData":
        # Read video
        if with_video:
            video = wbu.logged_artifacts(run, "video")[0]
            video = VideoArtifact.from_wandb_artifact(video)
            frames = video.read_frames()

        # Read config
        config: RunGenerativeRenderingConfig = OmegaConf.create(run.config)

        # get prompt
        prompt = config.prompt

        # get gr config
        gr_config = config.generative_rendering

        if with_anim:
            # read animation
            anim = wbu.used_artifacts(run, "animation")[0]
            anim = AnimationArtifact.from_wandb_artifact(anim)
            cams, meshes = anim.load_frames()
            verts_uvs, faces_uvs = anim.uv_data()

            # render depth maps and uv maps
            depths = render_depth_map(meshes, cams)
            uvs = render_rgb_uv_map(meshes, cams, verts_uvs, faces_uvs)

        return cls(
            frames=frames,
            prompt=prompt,
            gr_config=gr_config,
            cams=cams,
            meshes=meshes,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
            uvs=uvs,
            depths=depths,
        )
