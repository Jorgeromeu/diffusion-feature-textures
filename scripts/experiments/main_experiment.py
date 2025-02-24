import itertools
from dataclasses import dataclass
from typing import List

import numpy as np
from omegaconf import DictConfig, OmegaConf

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
from text3d2video.noise_initialization import UVNoiseInitializer
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.pipelines.reposable_diffusion_pipeline import ReposableDiffusionConfig
from text3d2video.rendering import render_depth_map
from text3d2video.utilities.video_comparison import group_into_array, video_grid
from text3d2video.utilities.video_util import (
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
        def scene_key(r):
            cfg = OmegaConf.create(r.config)
            anim: AnimationConfig = cfg.animation
            prompt = cfg.prompt
            return (prompt, anim.artifact_tag, anim.n_frames)

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

        runs_grouped = group_into_array(runs, dim_key_fns=[scene_key, type_key])
        return runs_grouped

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
