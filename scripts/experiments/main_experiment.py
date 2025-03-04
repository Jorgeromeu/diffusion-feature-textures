import itertools
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from omegaconf import DictConfig, OmegaConf
from sympy import per

import wandb_util.wandb_util as wbu
from scripts.run_generative_rendering import (
    ModelConfig,
    RunGenerativeRendering,
    RunGenerativeRenderingConfig,
)
from scripts.run_reposable_diffusion_t2v import (
    RunReposableDiffusionT2V,
    RunReposableDiffusionT2VConfig,
)
from text3d2video.artifacts.anim_artifact import AnimationArtifact, AnimationConfig
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.backprojection import project_visible_texels_to_camera
from text3d2video.noise_initialization import UVNoiseInitializer
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.pipelines.reposable_diffusion_pipeline import ReposableDiffusionConfig
from text3d2video.rendering import render_depth_map
from text3d2video.util import sample_feature_map_ndc
from text3d2video.utilities.video_comparison import group_into_array, video_grid
from text3d2video.utilities.video_util import (
    clip_to_pil_frames,
    extend_clip_to_match_duration,
    pil_frames_to_clip,
)
from wandb.apis.public import Run
from wandb_util.experiment_util import (
    object_to_instantiate_config,
)


@dataclass
class MainExperimentConfig:
    model: ModelConfig
    generative_rendering: GenerativeRenderingConfig
    reposable_diffusion: ReposableDiffusionConfig
    animations: list[AnimationConfig]
    prompts: list[str]


class MainExperiment(wbu.Experiment):
    experiment_name = "main_experiment"
    config: MainExperimentConfig

    def __init__(self, config: DictConfig):
        self.config = config

    def specification(self) -> List[wbu.RunDescriptor]:
        noise_initialization = object_to_instantiate_config(UVNoiseInitializer())

        runs = []

        run_config = wbu.RunConfig(
            wandb=True,
            instant_exit=False,
            download_artifacts=False,
            name=None,
            group=None,
            tags=[],
        )

        prompts = self.config.prompts
        animations = self.config.animations

        for prompt, animation in itertools.product(prompts, animations):
            gr_config = RunGenerativeRenderingConfig(
                run=run_config,
                prompt=prompt,
                animation=animation,
                generative_rendering=self.config.generative_rendering,
                model=self.config.model,
                noise_initialization=noise_initialization,
                seed=0,
            )

            rd_config = RunReposableDiffusionT2VConfig(
                run=run_config,
                prompt=prompt,
                animation=animation,
                reposable_diffusion=self.config.reposable_diffusion,
                model=self.config.model,
                noise_initialization=noise_initialization,
                num_views=10,
                seed=0,
            )

            per_frame_gr_config: GenerativeRenderingConfig = (
                self.config.generative_rendering.copy()
            )
            per_frame_gr_config.do_pre_attn_injection = False
            per_frame_gr_config.do_post_attn_injection = False
            per_frame_config = RunGenerativeRenderingConfig(
                run=run_config,
                prompt=prompt,
                animation=animation,
                generative_rendering=per_frame_gr_config,
                model=self.config.model,
                noise_initialization=noise_initialization,
                seed=0,
            )

            per_frame_run = wbu.RunDescriptor(
                RunGenerativeRendering(), OmegaConf.structured(per_frame_config)
            )

            gr_run = wbu.RunDescriptor(
                RunGenerativeRendering(), OmegaConf.structured(gr_config)
            )

            rd_run = wbu.RunDescriptor(
                RunReposableDiffusionT2V(), OmegaConf.structured(rd_config)
            )

            runs.append(gr_run)
            runs.append(rd_run)
            runs.append(per_frame_run)

        return runs

    # Processing:

    def get_depth_video(self, run):
        n_frames = OmegaConf.create(run.config).animation.n_frames

        vid = wbu.first_logged_artifact_of_type(run, "video")
        vid = VideoArtifact.from_wandb_artifact(vid)
        fps = vid.get_moviepy_clip().fps

        anim = wbu.first_used_artifact_of_type(run, "animation")
        anim = AnimationArtifact.from_wandb_artifact(anim)

        frame_indices = anim.frame_indices(n_frames)
        cams, meshes = anim.load_frames(frame_indices)
        depth_frames = render_depth_map(meshes, cams)
        return pil_frames_to_clip(depth_frames, fps)

    def get_grouped_runs(self):
        runs = self.get_logged_runs()

        # return scene
        def anim_key(r):
            cfg = OmegaConf.create(r.config)
            anim: AnimationConfig = cfg.animation
            return (anim.artifact_tag, anim.n_frames)

        def prompt_key(r):
            cfg = OmegaConf.create(r.config)
            return cfg.prompt

        # return run type
        def type_key(r):
            job_type = r.job_type

            if job_type == RunGenerativeRendering.job_type:
                cfg = OmegaConf.create(r.config)

                if not cfg.generative_rendering.do_pre_attn_injection:
                    return 0
                else:
                    return 1

            else:
                return 2

        runs_grouped = group_into_array(
            runs, dim_key_fns=[prompt_key, anim_key, type_key]
        )
        return runs_grouped

    def get_output_video(self, run):
        for art in run.logged_artifacts():
            if art.type == "video":
                return VideoArtifact.from_wandb_artifact(art)

    def get_output_videos(self, per_frame_run, gr_run, rd_run):
        # find aggr and video artifacts
        for art in rd_run.logged_artifacts():
            if art.type != "video":
                continue

            if art.name.startswith("video"):
                rd_video_art = art
            else:
                rd_aggr_art = art

        # find artifacts for gr and per frame
        gr_video_art = wbu.first_logged_artifact_of_type(gr_run, "video")
        per_frame_video_art = wbu.first_logged_artifact_of_type(per_frame_run, "video")

        # get videos
        video_artifacts = [per_frame_video_art, gr_video_art, rd_video_art, rd_aggr_art]

        videos = [
            VideoArtifact.from_wandb_artifact(art).get_moviepy_clip()
            for art in video_artifacts
        ]

        return videos

    def row_videos(self, runs):
        per_frame, gr, rd = runs
        depth_video = self.get_depth_video(gr)
        videos = self.get_output_videos(per_frame, gr, rd)

        vids = [depth_video] + videos
        vids = [extend_clip_to_match_duration(v, depth_video.duration) for v in vids]
        return vids

    def vid_comparison(self, grouped_runs, with_labels=False):
        videos = np.array([self.row_videos(runs) for runs in grouped_runs])

        labels = [
            "Geometry",
            "Per Frame",
            "Generative Rendering",
            "Ours (Target)",
            "Ours (Source)",
        ]

        if not with_labels:
            labels = None

        return video_grid(videos, col_gap_indices=[0, 1, 2], x_labels=labels)


class LoggedRun:
    def __init__(self, run):
        self.run = run

    @property
    def config(self):
        return OmegaConf.create(self.run.config)

    @property
    def job_type(self):
        return self.run.job_type

    @property
    def name(self):
        return self.run.name

    @property
    def url(self):
        return self.run.get_url()


class VideoGenerationRun(LoggedRun):
    def get_logged_video_art(self):
        vid_art = wbu.first_logged_artifact_of_type(self.run, "video")
        return VideoArtifact.from_wandb_artifact(vid_art)

    def get_input_anim_art(self):
        anim_art = wbu.first_used_artifact_of_type(self.run, "animation")
        return AnimationArtifact.from_wandb_artifact(anim_art)

    def get_frames_meshes_cams(self):
        """
        Get Input/Output frames and geometry
        """

        # get artifacts
        anim = self.get_input_anim_art()
        video = self.get_logged_video_art()

        # number of frames
        n_frames = self.config.animation.n_frames

        # get video, and frames
        clip = video.get_moviepy_clip()
        frames = clip_to_pil_frames(clip, expected_frames=n_frames)

        # get meshes and cams
        frame_indices = anim.frame_indices(n_frames)
        cams, meshes = anim.load_frames(frame_indices)

        return frames, meshes, cams


def get_uv_feature_maps(
    meshes, cams, feature_maps, verts_uvs, faces_uvs, texture_res=512
):
    assert len(meshes) == len(cams) == len(feature_maps)

    n_frames = len(meshes)

    textures = []

    for i in range(n_frames):
        mesh = meshes[i]
        cam = cams[i]
        feature_map = feature_maps[i]

        # compute inverse mapping
        xys, uvs = project_visible_texels_to_camera(
            mesh, cam, verts_uvs, faces_uvs, texture_res, raster_res=1200
        )
        xys = xys.cpu()
        uvs = uvs.cpu()

        # sample colors
        colors = sample_feature_map_ndc(feature_map, xys, mode="nearest")

        # populate texture
        texture = torch.zeros(texture_res, texture_res, feature_map.shape[0])
        texture[uvs[:, 1], uvs[:, 0], :] = colors
        textures.append(texture)

    return torch.stack(textures)


def uv_mse(cams, meshes, frames, verts_uvs, faces_uvs, texture_res=512):
    frames_pt = [TF.to_tensor(f) for f in frames]

    textures = get_uv_feature_maps(
        meshes, cams, frames_pt, verts_uvs, faces_uvs, texture_res
    )

    mses = []
    prev_tex = textures[0]
    for i in range(1, len(textures)):
        tex = textures[i]
        prev_mask = prev_tex.sum(dim=-1) > 0
        cur_mask = tex.sum(dim=-1) > 0
        mask = prev_mask & cur_mask
        prev_masked = prev_tex[mask]
        cur_masked = tex[mask]
        mse = F.mse_loss(prev_masked, cur_masked)
        mses.append(mse.item())

    mses = torch.tensor(mses)
    return torch.mean(mse)


def main_exp_video(groups: np.array):
    def get_clip(r: VideoGenerationRun):
        return VideoGenerationRun(r).get_logged_video_art().get_moviepy_clip

    clips = np.vectorize(get_clip)(groups)
    return video_grid(clips)
