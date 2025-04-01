from pathlib import Path
from typing import Dict, List

import torch
from attr import dataclass
from jaxtyping import Float
from PIL.Image import Image
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
)
from pytorch3d.structures import Meshes
from torch import Tensor
from tqdm import tqdm

from text3d2video.attn_processors.extraction_injection_attn import (
    UnifiedAttnProcessor,
)
from text3d2video.backprojection import (
    update_uv_texture,
)
from text3d2video.noise_initialization import UVNoiseInitializer
from text3d2video.pipelines.controlnet_pipeline import BaseControlNetPipeline
from text3d2video.rendering import (
    TextureShader,
    make_repeated_uv_texture,
    precompute_rasterization,
    render_depth_map,
)
from text3d2video.sd_feature_extraction import AttnLayerId
from text3d2video.util import dict_map
from text3d2video.utilities.logging import FeatureExtractionLogger


# pylint: disable=too-many-instance-attributes
@dataclass
class TexturingConfig:
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    module_paths: list[str]


@dataclass
class TexturingModelOutput:
    noise_preds: Float[Tensor, "b c h w"]
    extracted_kvs: Dict[str, Float[Tensor, "b t c"]]
    extracted_feats: Dict[str, Float[Tensor, "b c h w"]]


@dataclass
class TexturingGuidedModelOutput:
    noise_preds: Float[Tensor, "b c h w"]
    extracted_kvs_cond: Dict[str, Float[Tensor, "b t c"]]
    extracted_feats_cond: Dict[str, Float[Tensor, "b t c"]]
    extracted_kvs_uncond: Dict[str, Float[Tensor, "b t c"]]
    extracted_feats_uncond: Dict[str, Float[Tensor, "b c h w"]]


class TexturingPipeline(BaseControlNetPipeline):
    attn_processor: UnifiedAttnProcessor
    conf: TexturingConfig

    def model_forward_guided(
        self,
        latents: Float[Tensor, "b c h w"],
        t: int,
        cond_embeddings: Float[Tensor, "b t c"],
        uncond_embeddings: Float[Tensor, "b t c"],
        depth_maps: List[Image],
        injected_kvs_cond: Float[Tensor, "b t c"] = None,
        injected_kvs_uncond: Float[Tensor, "b t c"] = None,
        injected_feats_cond: Float[Tensor, "b c h w"] = None,
        injected_feats_uncond: Float[Tensor, "b c h w"] = None,
        extract_kvs: bool = False,
        extract_feats: bool = False,
    ) -> TexturingGuidedModelOutput:
        # perform multiple passes through model and compose noise predictions

        out_cond = self.model_forward(
            latents,
            t,
            cond_embeddings,
            depth_maps,
            # injected features
            injected_feats=injected_feats_cond,
            injected_kvs=injected_kvs_cond,
            # extraction settings
            extract_kvs=extract_kvs,
            extract_feats=extract_feats,
        )

        out_uncond = self.model_forward(
            latents,
            t,
            uncond_embeddings,
            depth_maps,
            # injected features
            injected_feats=injected_feats_uncond,
            injected_kvs=injected_kvs_uncond,
            # extraction settings
            extract_kvs=extract_kvs,
            extract_feats=extract_feats,
        )

        noise_cond = out_cond.noise_preds
        noise_uncond = out_uncond.noise_preds

        noise = noise_uncond + self.conf.guidance_scale * (noise_cond - noise_uncond)

        return TexturingGuidedModelOutput(
            noise,
            out_cond.extracted_kvs,
            out_cond.extracted_feats,
            out_uncond.extracted_kvs,
            out_uncond.extracted_feats,
        )

    def model_forward(
        self,
        latents,
        t,
        embeddings,
        depth_maps,
        injected_kvs=None,
        injected_feats=None,
        extract_kvs=False,
        extract_feats=False,
    ):
        # ControlNet pass
        processed_ctrl_images = self.preprocess_controlnet_images(depth_maps)
        down_residuals, mid_residual = self.controlnet(
            latents,
            t,
            encoder_hidden_states=embeddings,
            controlnet_cond=processed_ctrl_images,
            conditioning_scale=self.conf.controlnet_conditioning_scale,
            guess_mode=False,
            return_dict=False,
        )

        self.attn_processor.clear_injected_features()
        self.attn_processor.disable_all_extraction()
        self.attn_processor.do_kv_extraction = extract_kvs
        self.attn_processor.do_spatial_post_attn_extraction = extract_feats

        # pass injected features
        if injected_kvs is not None:
            self.attn_processor.injected_kvs = injected_kvs

        if injected_feats is not None:
            self.attn_processor.injected_post_attns = injected_feats

        # UNet pass with residuals
        noise_pred = self.unet(
            latents,
            t,
            mid_block_additional_residual=mid_residual,
            down_block_additional_residuals=down_residuals,
            encoder_hidden_states=embeddings,
        ).sample

        extracted_kvs = self.attn_processor.extracted_kvs
        extracted_feats = self.attn_processor.extracted_post_attns

        self.attn_processor.clear_extracted_features()

        return TexturingModelOutput(noise_pred, extracted_kvs, extracted_feats)

    @torch.no_grad()
    def __call__(
        self,
        prompts: str,
        meshes: Meshes,
        cameras: FoVPerspectiveCameras,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        conf: TexturingConfig,
        generator=None,
    ):
        self.conf = conf

        features_writer = FeatureExtractionLogger(Path("features.h5"))
        features_writer.delete_data()
        features_writer.open_write()

        # setup attn processor
        self.attn_processor = UnifiedAttnProcessor(
            self.unet,
            also_attend_to_self=True,
            feature_blend_alhpa=1.0,
            do_kv_extraction=True,
            do_spatial_post_attn_extraction=True,
            kv_extraction_paths=conf.module_paths,
            spatial_post_attn_extraction_paths=conf.module_paths,
        )
        self.unet.set_attn_processor(self.attn_processor)

        # number of images being generated
        batch_size = len(meshes)

        # render depth maps
        depth_maps = render_depth_map(meshes, cameras, 512)

        # precompute rasterization for all camera views and relevant resolutions
        layers = [AttnLayerId.parse(path) for path in self.conf.module_paths]
        screen_resolutions = list(
            set([layer.resolution(self.unet) for layer in layers])
        )
        uv_resolutions = [screen * 4 for screen in screen_resolutions]

        layer_resolution_indices = {
            layer.module_path(): screen_resolutions.index(layer.resolution(self.unet))
            for layer in layers
        }

        projections, fragments = precompute_rasterization(
            cameras,
            meshes,
            verts_uvs,
            faces_uvs,
            screen_resolutions,
            uv_resolutions,
        )

        # encode prompt
        cond_embeddings, uncond_embeddings = self.encode_prompt(prompts)

        # set timesteps
        self.scheduler.set_timesteps(conf.num_inference_steps)

        # initialize latents from standard normal
        noise_initializer = UVNoiseInitializer(
            noise_texture_res=200,
            device=self.device,
            dtype=self.dtype,
        )
        noise_initializer.sample_noise_texture(generator)
        noise_initializer.sample_background(generator)

        features_writer.write("uv_noise", noise_initializer.uv_noise)
        features_writer.write("bg_noise", noise_initializer.bg_noise)

        latents = noise_initializer.initial_noise(
            meshes,
            cameras,
            verts_uvs,
            faces_uvs,
        )

        for t in tqdm(self.scheduler.timesteps):
            # initialize empty textures
            def initial_texture(layer):
                dimension = layer.layer_channels(self.unet)
                res_idx = layer_resolution_indices[layer.module_path()]
                uv_res = uv_resolutions[res_idx]
                return torch.zeros(uv_res, uv_res, dimension).to(latents)

            textures_cond = {l.module_path(): initial_texture(l) for l in layers}
            textures_uncond = {l.module_path(): initial_texture(l) for l in layers}

            # initialize kvs
            prev_kvs_cond = None
            prev_kvs_uncond = None

            noise_preds = []
            for i in range(len(cameras)):
                # render feature textures
                def render_texture(layer, uv_map):
                    res_idx = layer_resolution_indices[layer]
                    frags = fragments[i][res_idx]

                    tex = make_repeated_uv_texture(
                        uv_map, faces_uvs, verts_uvs, sampling_mode="bilinear"
                    )

                    shader = TextureShader()

                    render_mesh = meshes[i].clone()
                    render_mesh.textures = tex
                    return shader(frags, render_mesh).to(self.dtype)

                # render current textures
                rendered_cond = dict_map(textures_cond, render_texture)
                rendered_uncond = dict_map(textures_uncond, render_texture)

                model_out = self.model_forward_guided(
                    latents[[i]],
                    t,
                    cond_embeddings[[i]],
                    uncond_embeddings[[i]],
                    [depth_maps[i]],
                    injected_feats_cond=rendered_cond,  # inject rendered feats
                    injected_feats_uncond=rendered_uncond,
                    injected_kvs_cond=prev_kvs_cond,  # inject first-view kvs
                    injected_kvs_uncond=prev_kvs_uncond,
                    extract_feats=True,
                    extract_kvs=True,
                )

                noise_pred = model_out.noise_preds[0]

                # update kvs
                extracted_kvs_cond = model_out.extracted_kvs_cond
                extracted_kvs_uncond = model_out.extracted_kvs_uncond

                prev_kvs_cond = extracted_kvs_cond
                prev_kvs_uncond = extracted_kvs_uncond

                # update feature textures
                extracted_cond = model_out.extracted_feats_cond
                extracted_uncond = model_out.extracted_feats_uncond

                for l in extracted_cond:
                    res_idx = layer_resolution_indices[l]
                    projection = projections[i][res_idx]

                    textures_cond[l] = torch.zeros_like(textures_cond[l])
                    update_uv_texture(
                        textures_cond[l],
                        extracted_cond[l][0],
                        projection.xys,
                        projection.uvs,
                    )

                    textures_uncond[l] = torch.zeros_like(textures_cond[l])
                    update_uv_texture(
                        textures_uncond[l],
                        extracted_uncond[l][0],
                        projection.xys,
                        projection.uvs,
                    )

                noise_preds.append(noise_pred)

            # save features
            features_writer.write_features_dict("tex", textures_cond, t=t, chunk="cond")
            features_writer.write_features_dict(
                "tex", textures_uncond, t=t, chunk="uncond"
            )

            def unsqueeze(layer, kvs):
                return kvs[0]

            # kvs_cond = dict_map(first_kvs_cond, unsqueeze)
            # kvs_uncond = dict_map(first_kvs_uncond, unsqueeze)
            # features_writer.write_features_dict("kvs", kvs_cond, t=t, chunk="cond")
            # features_writer.write_features_dict("kvs", kvs_uncond, t=t, chunk="uncond")

            noise_preds = torch.stack(noise_preds)
            latents = self.scheduler.step(noise_preds, t, latents).prev_sample

        # decode latents in chunks
        decoded_imgs = []
        # chunk indices to use in inference loop
        chunks_indices = torch.split(torch.arange(0, batch_size), 5)
        for chunk_frame_indices in chunks_indices:
            chunk_latents = latents[chunk_frame_indices]
            chunk_images = self.decode_latents(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        return decoded_imgs
