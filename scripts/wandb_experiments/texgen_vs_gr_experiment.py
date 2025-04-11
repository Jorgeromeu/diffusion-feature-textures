from typing import List

import torch
import torchvision.transforms.functional as TF
from attr import dataclass
from omegaconf import OmegaConf
from PIL.Image import Image

import wandb_util.wandb_util as wbu
from scripts.wandb_runs.make_rgb_texture import MakeTexture, MakeTextureConfig
from scripts.wandb_runs.run_generative_rendering import (
    RunGenerativeRendering,
    RunGenerativeRenderingConfig,
)
from scripts.wandb_runs.run_texgen import RunTexGen, RunTexGenConfig
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.artifacts.texture_artifact import TextureArtifact
from text3d2video.artifacts.video_artifact import VideoArtifact
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
)
from text3d2video.pipelines.pipeline_utils import ModelConfig
from text3d2video.pipelines.texgen_pipeline import TexGenConfig
from text3d2video.rendering import render_rgb_uv_map, render_texture
from text3d2video.utilities.video_comparison import video_grid
from text3d2video.utilities.video_util import pil_frames_to_clip
from wandb import controller


@dataclass
class LoggedData:
    gr_frames: List[Image]
    controlnet_frames: List[Image]
    renders: List[Image]
    texture: torch.Tensor
    uvs: List[Image]


class TexGenVsGrExperiment(wbu.Experiment):
    experiment_name = "texgen_vs_gr"

    def specification(self):
        prompt = "Stormtrooper"
        anim_tag = "anim:latest"
        texturing_tag = "human_mv:latest"
        seed = 0

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
            do_pre_attn_injection=True,
            do_post_attn_injection=True,
            feature_blend_alpha=1.0,
            attend_to_self_kv=False,
            mean_features_weight=0.5,
            chunk_size=5,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            module_paths=decoder_paths,
            num_keyframes=1,
            num_inference_steps=10,
        )

        controlnet_config = GenerativeRenderingConfig(
            do_pre_attn_injection=False,
            do_post_attn_injection=False,
            feature_blend_alpha=1.0,
            attend_to_self_kv=False,
            mean_features_weight=0.5,
            chunk_size=5,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            module_paths=decoder_paths,
            num_keyframes=1,
            num_inference_steps=10,
        )

        texgen_config = TexGenConfig(
            num_inference_steps=10,
            guidance_scale=7.5,
            controlnet_conditioning_scale=1.0,
            module_paths=decoder_paths,
            quality_update_factor=1.1,
            uv_res=512,
        )

        model_config = ModelConfig(
            sd_repo="runwayml/stable-diffusion-v1-5",
            controlnet_repo="lllyasviel/control_v11f1p_sd15_depth",
        )

        # Create GR Run
        run_gr_config = RunGenerativeRenderingConfig(
            prompt=prompt,
            animation_tag=anim_tag,
            generative_rendering=gr_config,
            model=model_config,
            seed=seed,
        )
        run_gr_config = OmegaConf.structured(run_gr_config)
        run_gr = wbu.RunSpecification("GR", RunGenerativeRendering(), run_gr_config)

        # Create ControlNet Run
        run_controlnet_config = RunGenerativeRenderingConfig(
            prompt=prompt,
            animation_tag=anim_tag,
            generative_rendering=controlnet_config,
            model=model_config,
            seed=seed,
        )
        run_controlnet_config = OmegaConf.structured(run_controlnet_config)
        run_controlnet = wbu.RunSpecification(
            "ControlNet", RunGenerativeRendering(), run_controlnet_config
        )

        # create TexGen Run
        run_texgen_config = RunTexGenConfig(
            prompt=prompt,
            animation_tag=texturing_tag,
            texgen=texgen_config,
            model=model_config,
            out_art_name="texture_views",
            seed=seed,
        )

        run_texgen_config = OmegaConf.structured(run_texgen_config)
        run_texgen = wbu.RunSpecification("TexGen", RunTexGen(), run_texgen_config)

        # create Make Texture run
        create_texure_config = MakeTextureConfig(
            video_anim="texture_views:latest",
        )
        create_texure_config = OmegaConf.structured(create_texure_config)
        create_texture = wbu.RunSpecification(
            "make_texture", MakeTexture(), create_texure_config, depends_on=[run_texgen]
        )

        return [
            run_gr,
            run_controlnet,
            run_texgen,
            create_texture,
        ]

    def get_data(self):
        logged_runs = self.get_logged_runs()

        # get runs
        for r in logged_runs:
            if r.name == "GR":
                gr_run = r
            if r.name == "TexGen":
                texgen_run = r
            if r.name == "ControlNet":
                controlnet_run = r
            if r.name == "make_texture":
                make_texture_run = r

        # get gr frames
        gr_vid = wbu.logged_artifacts(gr_run, type="video")[0]
        gr_vid = VideoArtifact.from_wandb_artifact(gr_vid)
        gr_frames = gr_vid.read_frames()

        # get ControlNet frames
        controlnet_vid = wbu.logged_artifacts(controlnet_run, type="video")[0]
        controlnet_vid = VideoArtifact.from_wandb_artifact(controlnet_vid)
        controlnet_frames = controlnet_vid.read_frames()

        # get texture
        texture = wbu.logged_artifacts(make_texture_run, type="rgb_texture")
        texture = TextureArtifact.from_wandb_artifact(texture[0])
        texture = texture.read_texture()

        # read animation
        anim = wbu.used_artifacts(gr_run, type="animation")[0]
        anim = AnimationArtifact.from_wandb_artifact(anim)
        cams, meshes = anim.load_frames()
        anim_verts_uvs, anim_faces_uvs = anim.uv_data()

        # render uvs
        uvs = render_rgb_uv_map(meshes, cams, anim_verts_uvs, anim_faces_uvs)

        # render texture
        with torch.no_grad():
            # render texture
            renders = render_texture(
                meshes, cams, texture, anim_verts_uvs, anim_faces_uvs
            ).cpu()
        renders = [TF.to_pil_image(r) for r in renders]

        return LoggedData(
            gr_frames=gr_frames,
            renders=renders,
            controlnet_frames=controlnet_frames,
            uvs=uvs,
            texture=texture,
        )

    def comparison_vid(self, data=None):
        if data is None:
            data = self.get_data()

        gr_vid = pil_frames_to_clip(data.gr_frames)
        controlnet_vid = pil_frames_to_clip(data.controlnet_frames)
        texture_vid = pil_frames_to_clip(data.renders)
        uv_vid = pil_frames_to_clip(data.uvs)

        videos = [
            (uv_vid, "UV"),
            (controlnet_vid, "ControlNet"),
            (gr_vid, "Generative Rendering"),
            (texture_vid, "Static Texture"),
        ]

        vids = [v[0] for v in videos]
        labels = [v[1] for v in videos]

        return video_grid(
            [vids],
            x_labels=labels,
        )
