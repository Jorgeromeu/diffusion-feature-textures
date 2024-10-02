from math import sqrt
from typing import List

import torch
from diffusers import StableDiffusionControlNetPipeline
from PIL.Image import Image
import hydra
from omegaconf import DictConfig
from text3d2video.artifacts.animation_artifact import AnimationArtifact
from text3d2video.artifacts.multiview_features_artifact import MVFeaturesArtifact
from text3d2video.artifacts.vertex_atributes_artifact import VertAttributesArtifact
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.cross_frame_attn import CrossFrameAttnProcessor
from text3d2video.multidict import MultiDict
from text3d2video.pipelines.my_pipeline import MyPipeline
from text3d2video.rendering import rasterize_vertex_features, render_depth_map
import text3d2video.wandb_util as wu
import wandb

from text3d2video.util import front_camera
from diffusers import ControlNetModel


def render_feature_images(
    vert_features: VertAttributesArtifact,
    mv_features: MVFeaturesArtifact,
    animation: AnimationArtifact,
    frame_indices: List[int],
    timesteps: List[int],
) -> MultiDict:

    # read vertex features to multidict
    vert_features_multidict = vert_features.get_disk_multidict()

    # store feature images here
    all_feature_images = MultiDict()

    # setup camera and frames
    camera = front_camera()
    frames = animation.load_frames(frame_indices)

    for identifier in vert_features_multidict.keys():

        if int(identifier["timestep"]) not in timesteps:
            continue

        vert_features = vert_features_multidict[identifier].cuda()

        # feature resolution
        shape = mv_features.get_feature_shape(identifier)
        feature_res = int(sqrt(shape[0]))

        # rasterize vertex features
        feature_images = torch.stack(
            [
                rasterize_vertex_features(camera, frame, feature_res, vert_features)
                for frame in frames
            ]
        )

        all_feature_images[identifier] = feature_images

    return all_feature_images


@hydra.main(config_path="../config", config_name="config")
def run(cfg: DictConfig):

    video_cfg = cfg.generate_video

    # init wandb
    wu.init_run(dev_run=cfg.dev_run, job_type="multiview_features")
    wandb.config.update(dict(video_cfg))

    # setup pipeline
    dtype = torch.float16
    sd_repo = cfg.stable_diffusion.name
    controlnet_repo = cfg.controlnet.name
    device = torch.device("cuda")

    controlnet = ControlNetModel.from_pretrained(
        controlnet_repo, torch_dtype=torch.float16
    ).to(device)

    pipe = MyPipeline.from_pretrained(
        sd_repo, controlnet=controlnet, torch_dtype=dtype
    ).to(device)

    # read animation
    animation = wu.get_artifact(video_cfg.animation_artifact_tag)
    animation = AnimationArtifact.from_wandb_artifact(animation)

    # read 3D features
    features_3d = wu.get_artifact(video_cfg.vertex_attributes_artifact_tag)
    features_3d = VertAttributesArtifact.from_wandb_artifact(features_3d)

    # get mv features from lineage
    mv_features = features_3d.get_mv_features_from_lineage()

    # setup camera and frames
    camera = front_camera()
    frames = animation.load_frames(video_cfg.frame_indices)

    # render depth maps
    depth_maps = render_depth_map(frames, camera, video_cfg.out_resolution)

    # render feature images
    all_feature_images = render_feature_images(
        features_3d,
        mv_features,
        animation,
        video_cfg.frame_indices,
    )

    # Attention Processor setup
    attn_processor = CrossFrameAttnProcessor(unet_chunk_size=2, unet=pipe.unet)
    attn_processor.feature_images_multidict = all_feature_images
    attn_processor.do_cross_frame_attn = True
    attn_processor.do_feature_injection = True
    pipe.unet.set_attn_processor(attn_processor)

    # run pipeline
    prompts = [video_cfg.prompt] * len(video_cfg.frame_indices)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(video_cfg.seed)
    frames = pipe(prompts, depth_maps, generator=generator, num_inference_steps=30)

    video_artifact = VideoArtifact.create_wandb_artifact(
        video_cfg.out_artifact_name, frames=frames, fps=video_cfg.fps
    )
    wu.log_artifact_if_enabled(video_artifact)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run()
