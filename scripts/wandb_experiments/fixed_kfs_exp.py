from typing import List

from attr import dataclass
from omegaconf import OmegaConf
from PIL.Image import Image

import wandb_util.wandb_util as wbu
from scripts.wandb_runs.run_generative_rendering import (
    RunGenerativeRenderingConfig,
    run_generative_rendering,
)
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.pipelines.pipeline_utils import ModelConfig
from text3d2video.utilities.omegaconf_util import omegaconf_from_dotdict
from text3d2video.utilities.video_comparison import video_grid
from text3d2video.utilities.video_util import pil_frames_to_clip


@dataclass
class FixGrKeyframesExp:
    prompt: str
    anim_tag: str
    kf_indices: List[List[int]]


def fixed_gr_keyframes_exp(config: FixGrKeyframesExp):
    spec = []

    decoder_paths = [
        "mid_block.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.1.attentions.2.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.2.attentions.2.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.0.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.1.transformer_blocks.0.attn1",
        "up_blocks.3.attentions.2.transformer_blocks.0.attn1",
    ]

    gr_config = GenerativeRenderingConfig(
        module_paths=decoder_paths, num_inference_steps=15
    )

    run_gr = RunGenerativeRenderingConfig(
        config.prompt,
        config.anim_tag,
        gr_config,
        ModelConfig(),
    )
    run_gr = OmegaConf.structured(run_gr)

    # base GR run
    run_gr_spec = wbu.RunSpec("GR", run_generative_rendering, run_gr)
    spec += [run_gr_spec]

    # override ControlNet
    overrides = {
        "generative_rendering.do_pre_attn_injection": False,
        "generative_rendering.do_post_attn_injection": False,
    }
    overrides = omegaconf_from_dotdict(overrides)
    overriden_kfs = OmegaConf.merge(run_gr, overrides)
    controlnet = wbu.RunSpec("ControlNet", run_generative_rendering, overriden_kfs)
    spec += [controlnet]

    for kf_indices in config.kf_indices:
        # override keyframes
        overrides = {"generative_rendering.kf_indices": kf_indices}
        overrides = omegaconf_from_dotdict(overrides)
        overriden_kfs = OmegaConf.merge(run_gr, overrides)
        run_gr_spec_overriden_kfs = wbu.RunSpec(
            f"overriden_{kf_indices}", run_generative_rendering, overriden_kfs
        )
        spec += [run_gr_spec_overriden_kfs]

    return spec


@dataclass
class FixedKeyframeData:
    frames: List[Image]
    kf_indices: List[int]


@dataclass
class ExpData:
    controlnet_frames: List[Image]
    gr_frames: List[Image]
    fixed_kf_data: List[FixedKeyframeData]


def get_data(name):
    def get_frames(run):
        video = wbu.logged_artifacts(run, "video")[0]
        video = VideoArtifact.from_wandb_artifact(video)
        return video.read_frames()

    # read experiment
    runs = wbu.get_logged_runs(name)

    # categorize runs
    override_runs = []
    for r in runs:
        if r.name == "GR":
            gr = r
        elif r.name == "ControlNet":
            controlnet = r
        else:
            override_runs.append(r)

    controlnet_frames = get_frames(controlnet)
    gr_frames = get_frames(gr)

    override_data = []
    for r in override_runs:
        config = OmegaConf.create(r.config)
        kf_indices = config.generative_rendering.kf_indices
        frames = get_frames(r)
        data = FixedKeyframeData(frames, kf_indices)
        override_data.append(data)

    return ExpData(
        controlnet_frames=controlnet_frames,
        gr_frames=gr_frames,
        fixed_kf_data=override_data,
    )


def make_video(name, data=None):
    if data is None:
        data = get_data(name)

    controlnet_vid = pil_frames_to_clip(data.controlnet_frames)
    gr_vid = pil_frames_to_clip(data.gr_frames)

    videos = [controlnet_vid, gr_vid]
    titles = ["ControlNet", "Generative Rendering"]

    for r in data.fixed_kf_data:
        kf_frames = [data.controlnet_frames[i] for i in r.kf_indices]
        kf_vid = pil_frames_to_clip(kf_frames, fps=2)
        vid = pil_frames_to_clip(r.frames)

        videos.append(kf_vid)
        videos.append(vid)
        titles.append("keyframes")
        titles.append("Fixed Keyframes")

    gap_indices = [0, 1]
    for i in range(len(data.fixed_kf_data)):
        last_index = gap_indices[-1]
        gap_indices.append(last_index + 2)

    comparison_vid = video_grid([videos], col_gap_indices=gap_indices, x_labels=titles)
    return comparison_vid
