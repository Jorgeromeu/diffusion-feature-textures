import itertools
from math import sqrt
from typing import List

from codetiming import Timer
from einops import rearrange
import torch
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from text3d2video.artifacts.animation_artifact import AnimationArtifact
from text3d2video.artifacts.multiview_features_artifact import MVFeaturesArtifact
from text3d2video.artifacts.vertex_atributes_artifact import VertAttributesArtifact
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.cross_frame_attn import CrossFrameAttnProcessor
from text3d2video.diffusion import make_controlnet_diffusion_pipeline
from text3d2video.multidict import MultiDict
from text3d2video.pipelines.my_pipeline import MyPipeline
from text3d2video.rendering import (
    make_feature_renderer,
    rasterize_vertex_features,
    render_depth_map,
)
from pytorch3d.structures import join_meshes_as_batch
import text3d2video.wandb_util as wu
import wandb
from pytorch3d.renderer import TexturesVertex
from pytorch3d.structures import Meshes

from text3d2video.util import front_camera
from diffusers import ControlNetModel


def render_feature_images(
    vert_features: VertAttributesArtifact,
    mv_features: MVFeaturesArtifact,
    animation: AnimationArtifact,
    frame_indices: List[int],
    timesteps: List[int],
    layers: List[str],
) -> MultiDict:

    # read vertex features to multidict
    vert_features_multidict = vert_features.get_features_disk_dict()

    # store feature images here
    all_feature_images = MultiDict()

    # setup camera and frames
    camera = front_camera()
    frames = animation.load_frames(frame_indices)

    for layer, timestep in tqdm(itertools.product(layers, timesteps)):

        identifier = {"layer": layer, "timestep": timestep}

        vert_features = vert_features_multidict[identifier].cuda()

        batched_vert_features = vert_features.expand(len(frames), *vert_features.shape)
        vert_tex = TexturesVertex(verts_features=batched_vert_features)
        frames.textures = vert_tex

        # feature resolution
        shape = mv_features.get_feature_shape(layer)
        feature_res = int(sqrt(shape[0]))

        renderer = make_feature_renderer(camera, feature_res)
        feature_images = renderer(frames).detach().cpu()
        feature_images = rearrange(feature_images, "b h w d -> b d h w")

        all_feature_images[identifier] = feature_images

    return all_feature_images


@hydra.main(config_path="../config", config_name="config")
def run(cfg: DictConfig):

    video_cfg = cfg.generate_video

    # init wandb
    wu.init_run(dev_run=cfg.dev_run, job_type="generate_video", tags=cfg.wandb.tags)
    wandb.config.update(dict(video_cfg))
    wandb.config.upda

    # setup pipeline
    pipe = make_controlnet_diffusion_pipeline(
        cfg.model.sd_repo, cfg.model.controlnet_repo
    )

    # read animation
    animation = AnimationArtifact.from_wandb_artifact_tag(
        video_cfg.animation_artifact_tag,
        download=cfg.download_artifacts,
    )

    # read 3D features
    features_3d = VertAttributesArtifact.from_wandb_artifact_tag(
        video_cfg.vertex_attributes_artifact_tag, download=cfg.download_artifacts
    )

    # get mv features from lineage
    mv_features = features_3d.get_mv_features_from_lineage()

    # setup camera and frames
    camera = front_camera()
    frames = animation.load_frames(video_cfg.frame_indices)

    # render depth maps
    with torch.no_grad():
        depth_maps = render_depth_map(frames, camera, video_cfg.out_resolution)

    do_feature_injection = video_cfg.feature_blend_alpha > 0

    # render feature images
    if do_feature_injection:
        with Timer(text="rendering feature images: {}"):
            all_feature_images = render_feature_images(
                features_3d,
                mv_features,
                animation,
                video_cfg.frame_indices,
                timesteps=video_cfg.timesteps,
                layers=video_cfg.layers,
            )
            print("num feature images:", len(all_feature_images))
    else:
        all_feature_images = MultiDict()

    # setup attention processor
    attn_processor = CrossFrameAttnProcessor(unet_chunk_size=2, pipe=pipe)
    attn_processor.feature_images_multidict = all_feature_images
    attn_processor.do_cross_frame_attn = video_cfg.do_cross_frame_attention
    attn_processor.do_feature_injection = do_feature_injection
    attn_processor.feature_blend_alpha = video_cfg.feature_blend_alpha
    pipe.unet.set_attn_processor(attn_processor)

    # run pipeline
    prompt = cfg.inputs.animation_prompt
    prompts = [prompt] * len(video_cfg.frame_indices)
    generator = torch.Generator(device="cuda")
    generator.manual_seed(cfg.inputs.sd_seed)
    frames = pipe(prompts, depth_maps, generator=generator, num_inference_steps=30)

    # save video
    video_artifact = VideoArtifact.create_empty_artifact(video_cfg.out_artifact_name)
    video_artifact.write_frames(frames, fps=video_cfg.fps)

    # log video to run
    wandb.log({"video": wandb.Video(str(video_artifact.get_mp4_path()))})

    # save video artifact
    video_artifact.log()
    wandb.finish()


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    run()
