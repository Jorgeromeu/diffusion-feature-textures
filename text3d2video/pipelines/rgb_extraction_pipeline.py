from collections import defaultdict
from typing import Dict, List, Optional

import torch
from attr import dataclass
from jaxtyping import Float
from PIL import Image
from pytorch3d.renderer.mesh.rasterizer import Fragments
from torch import Generator, Tensor
from tqdm import tqdm

from text3d2video.artifacts.anim_artifact import AnimSequence
from text3d2video.backprojection import (
    TexelProjection,
    compute_texel_projection_old,
)
from text3d2video.pipelines.controlnet_pipeline import BaseControlNetPipeline
from text3d2video.pipelines.generative_rendering_pipeline import GrLogic
from text3d2video.rendering import (
    TextureShader,
    make_mesh_rasterizer,
    make_repeated_uv_texture,
    shade_meshes,
)
from text3d2video.util import dict_map


# pylint: disable=too-many-instance-attributes
@dataclass
class GenerativeRenderingConfig:
    do_pre_attn_injection: bool = True
    do_post_attn_injection: bool = True
    attend_to_self_kv: bool = True
    mean_features_weight: float = 0.5
    chunk_size: int = 5
    num_inference_steps: int = 15
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    num_keyframes: int = 3
    feature_blend_alpha: float = 0.5


@dataclass
class GrExtractedFeatures:
    cond_kv: Dict[str, Float[torch.Tensor, "b t d"]]
    uncond_kv: Dict[str, Float[torch.Tensor, "b t d"]]
    cond_post_attn: Dict[str, Float[torch.Tensor, "b f t d"]]
    uncond_post_attn: Dict[str, Float[torch.Tensor, "b f t d"]]

    def layer_resolution(self, layer):
        feature = self.cond_post_attn.get(layer, None)
        if feature is not None:
            return feature.shape[-1]
        else:
            return None


@dataclass
class GrCamSeq:
    seq: AnimSequence
    projections: List[List[TexelProjection]]
    fragments: List[List[Fragments]]
    depth_maps: List[Image.Image]


def precompute_rasterization(
    cameras, meshes, vert_uvs, faces_uvs, render_resolutions, texture_resolutions
):
    projections = defaultdict(lambda: dict())
    fragments = defaultdict(lambda: dict())

    for frame_idx in range(len(cameras)):
        cam = cameras[frame_idx]
        mesh = meshes[frame_idx]

        for res_i in range(len(render_resolutions)):
            render_res = render_resolutions[res_i]
            texture_res = texture_resolutions[res_i]

            # project UVs to camera
            projection = compute_texel_projection_old(
                mesh,
                cam,
                vert_uvs,
                faces_uvs,
                raster_res=texture_res * 10,
                texture_res=texture_res,
            )

            # rasterize
            rasterizer = make_mesh_rasterizer(
                resolution=render_res,
                faces_per_pixel=1,
                blur_radius=0,
                bin_size=0,
            )
            frame_fragments = rasterizer(mesh, cameras=cam)

            fragments[frame_idx][res_i] = frame_fragments
            projections[frame_idx][res_i] = projection

    return projections, fragments


class RgbExtractionPipeline(BaseControlNetPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        anim: AnimSequence,
        src_anim: AnimSequence,
        conf: GenerativeRenderingConfig,
        inversion_trajectory: Tensor,
        generator: Optional[Generator] = None,
    ):
        n_frames = len(anim.cams)

        # setup GR logic
        gr = GrLogic.from_gr_config(self, conf)
        gr.set_attn_processor()

        layer_resolutions, layer_resolution_indices = gr.calculate_layer_resolutions()
        uv_resolutions = [4 * res for res in layer_resolutions]
        frags_anim = gr.precompute_seq(anim, layer_resolutions, uv_resolutions)

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt] * n_frames)

        # chunk indices to use i regeneration
        chunks_indices = torch.split(torch.arange(0, n_frames), conf.chunk_size)

        latents = self.prepare_latents(n_frames)

        self.scheduler.set_timesteps(conf.num_inference_steps)

        # extr embeddings
        extr_cond_embs, extr_uncond_embs = self.encode_prompt([prompt] * len(src_anim))
        extr_depths = src_anim.render_depth_maps()

        frags_anim = gr.precompute_seq(anim, layer_resolutions, uv_resolutions)
        frags_src = gr.precompute_seq(src_anim, layer_resolutions, uv_resolutions)

        # denoising loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            inv_latents = inversion_trajectory[i]

            # perform feature extraction on inverted part
            feats = gr.model_forward_extraction(
                inv_latents,
                extr_cond_embs,
                extr_uncond_embs,
                extr_depths,
                t,
            )

            extr_projections = list(frags_src.projections.values())

            def aggr_feats(layer, features):
                res_idx = layer_resolution_indices[layer]
                uv_res = uv_resolutions[res_idx]
                projections = [projs[res_idx] for projs in extr_projections]
                return gr.aggregate_features(features, uv_res, projections)

            textures_uncond = dict_map(feats.uncond_post_attn, aggr_feats)
            textures_cond = dict_map(feats.cond_post_attn, aggr_feats)

            # denoise anim latents
            noise_preds = []
            for chunk_frame_indices in chunks_indices:
                # render chunk feats
                chunk_frags = [
                    frags_anim.fragments[i] for i in chunk_frame_indices.tolist()
                ]

                def render_chunk(layer, uv_map):
                    chunk_meshes = anim.meshes[chunk_frame_indices]
                    res_idx = layer_resolution_indices[layer]
                    frags = [f[res_idx] for f in chunk_frags]
                    texture = make_repeated_uv_texture(
                        uv_map,
                        anim.faces_uvs,
                        anim.verts_uvs,
                        sampling_mode="nearest",
                        N=1,
                    )
                    shader = TextureShader()
                    return shade_meshes(shader, texture, chunk_meshes, frags)

                renders_uncond = dict_map(textures_uncond, render_chunk)
                renders_cond = dict_map(textures_cond, render_chunk)

                # model forward with pre-and post-attn injection
                noise_pred = gr.model_forward_injection(
                    latents[chunk_frame_indices],
                    cond_embeddings[chunk_frame_indices],
                    uncond_embeddings[chunk_frame_indices],
                    [frags_anim.depth_maps[i] for i in chunk_frame_indices.tolist()],
                    t,
                    feats.cond_kv,
                    feats.uncond_kv,
                    {},
                    {},
                )

                noise_preds.append(noise_pred)

            noise_preds = torch.cat(noise_preds, dim=0)
            latents = self.scheduler.step(
                noise_preds, t, latents, generator=generator
            ).prev_sample

        return self.decode_latents(
            latents, chunk_size=conf.chunk_size, generator=generator
        )
