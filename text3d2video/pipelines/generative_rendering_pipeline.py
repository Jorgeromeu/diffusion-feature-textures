from typing import Dict, List, Optional

import torch
from attr import dataclass
from jaxtyping import Float
from PIL import Image
from pytorch3d.renderer.mesh.rasterizer import Fragments
from torch import Generator, Tensor
from tqdm import tqdm

from text3d2video.artifacts.anim_artifact import AnimSequence
from text3d2video.attn_processors.extraction_injection_attn import (
    ExtractionInjectionAttn,
)
from text3d2video.backprojection import (
    TexelProjection,
    aggregate_views_uv_texture,
    aggregate_views_uv_texture_mean,
)
from text3d2video.noise_initialization import (
    UVNoiseInitializer,
)
from text3d2video.pipelines.controlnet_pipeline import BaseControlNetPipeline
from text3d2video.rendering import (
    TextureShader,
    make_repeated_uv_texture,
    precompute_rasterization,
    render_depth_map,
    render_texture,
    shade_meshes,
)
from text3d2video.sd_feature_extraction import AttnLayerId
from text3d2video.util import dict_map, linear_map
from text3d2video.utilities.logging import (
    NULL_LOGGER,
    setup_greenlists,
    write_feature_dict,
    write_feature_frame_dict,
)


# pylint: disable=too-many-instance-attributes
@dataclass
class GenerativeRenderingConfig:
    module_paths: list[str]
    do_pre_attn_injection: bool = True
    do_post_attn_injection: bool = True
    feature_blend_alpha: float = 0.8
    attend_to_self_kv: bool = False
    mean_features_weight: float = 0.5
    chunk_size: int = 5
    num_inference_steps: int = 15
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    num_keyframes: int = 3


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


class GenerativeRenderingLogic:
    def __init__(
        self,
        pipe: BaseControlNetPipeline,
        module_paths: list[str] = [],
        guidance_scale=7.5,
        controlnet_conditioning_scale: float = 1.0,
        mean_features_weight: float = 0.5,
        do_pre_attn_injection: bool = True,
        do_post_attn_injection: bool = True,
    ):
        self.pipe = pipe
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.guidance_scale = guidance_scale
        self.mean_features_weight = mean_features_weight
        self.module_paths = module_paths
        # create attn processor
        self.attn_processor = ExtractionInjectionAttn(
            self.pipe.unet,
            do_spatial_post_attn_extraction=do_post_attn_injection,
            do_kv_extraction=do_pre_attn_injection,
            also_attend_to_self=False,
            feature_blend_alpha=0.5,
            kv_extraction_paths=module_paths,
            spatial_post_attn_extraction_paths=module_paths,
        )

    def set_attn_processor(self):
        self.pipe.unet.set_attn_processor(self.attn_processor)

    def model_forward(
        self,
        latents: Float[Tensor, "b c h w"],
        embeddings: Float[Tensor, "b t d"],
        t: int,
        depth_maps: List[Image.Image],
    ) -> Tensor:
        # ControlNet Pass
        processed_ctrl_images = self.pipe.preprocess_controlnet_images(depth_maps)
        down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
            latents,
            t,
            encoder_hidden_states=embeddings,
            controlnet_cond=processed_ctrl_images,
            conditioning_scale=self.controlnet_conditioning_scale,
            guess_mode=False,
            return_dict=False,
        )

        # UNet Pass
        noise_pred = self.pipe.unet(
            latents,
            t,
            mid_block_additional_residual=mid_block_res_sample,
            down_block_additional_residuals=down_block_res_samples,
            encoder_hidden_states=embeddings,
        ).sample

        return noise_pred

    def model_forward_extraction(
        self,
        latents: Float[Tensor, "b c h w"],
        cond_embeddings: Float[Tensor, "t d"],
        uncond_embeddings: Float[Tensor, "t d"],
        depth_maps: List[Image.Image],
        t: int,
    ) -> GrExtractedFeatures:
        # do cond and uncond passes
        latents_duplicated = torch.cat([latents] * 2)
        both_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        depth_maps_duplicated = depth_maps * 2

        # model pass, to extract features
        self.attn_processor.set_extraction_mode()
        self.attn_processor.set_chunk_labels(["uncond", "cond"])
        _ = self.model_forward(
            latents_duplicated, both_embeddings, t, depth_maps_duplicated
        )

        extracted_kv = self.attn_processor.kv_features
        extracted_post_attn = self.attn_processor.spatial_post_attn_features

        self.attn_processor.clear_features()

        extracted_kv_uncond = {}
        extracted_kv_cond = {}
        for layer in extracted_kv.keys():
            kvs = extracted_kv[layer]
            extracted_kv_uncond[layer] = kvs[0]
            extracted_kv_cond[layer] = kvs[1]

        extracted_post_attn_uncond = {}
        extracted_post_attn_cond = {}
        for layer in extracted_post_attn.keys():
            post_attns = extracted_post_attn[layer]
            n_frames = len(latents)
            extracted_post_attn_uncond[layer] = post_attns[:n_frames]
            extracted_post_attn_cond[layer] = post_attns[n_frames:]

        return GrExtractedFeatures(
            cond_kv=extracted_kv_cond,
            uncond_kv=extracted_kv_uncond,
            cond_post_attn=extracted_post_attn_cond,
            uncond_post_attn=extracted_post_attn_uncond,
        )

    def model_forward_injection(
        self,
        latents: Float[Tensor, "b c h w"],
        cond_embeddings: Float[Tensor, "t d"],
        uncond_embeddings: Float[Tensor, "t d"],
        depth_maps: List[Image.Image],
        t: int,
        cond_kv_features: Dict[str, Float[Tensor, "b t d"]],
        uncond_kv_features: Dict[str, Float[Tensor, "b t d"]],
        cond_rendered_features: Dict[str, Float[Tensor, "b f t d"]],
        uncond_rendered_features: Dict[str, Float[Tensor, "b f t d"]],
        progress: float = 1.0,
    ) -> Tensor:
        latents_duplicated = torch.cat([latents] * 2)
        embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        depth_maps_duplicated = depth_maps * 2

        # stack kv features
        injected_kvs = {}
        for layer in cond_kv_features.keys():
            layer_kvs = torch.stack(
                [uncond_kv_features[layer], cond_kv_features[layer]]
            )
            injected_kvs[layer] = layer_kvs

        # stack rendered features
        injected_post_attn = {}
        for layer in cond_rendered_features.keys():
            layer_post_attn = torch.stack(
                [uncond_rendered_features[layer], cond_rendered_features[layer]]
            )
            injected_post_attn[layer] = layer_post_attn

        # pass injected features
        self.attn_processor.set_injection_mode(
            pre_attn_features=injected_kvs, post_attn_features=injected_post_attn
        )

        noise_pred = self.model_forward(
            latents_duplicated, embeddings, t, depth_maps_duplicated
        )

        self.attn_processor.feature_blend_alpha = progress

        # classifier-free guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred_guided = noise_pred_uncond + self.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        return noise_pred_guided

    def aggregate_features(
        self,
        feature_maps: Float[Tensor, "b c h w"],
        uv_res: int,
        projections: List[TexelProjection],
    ) -> Float[Tensor, "h w c"]:
        texture = aggregate_views_uv_texture(
            feature_maps,
            uv_res,
            projections,
            interpolation_mode="bilinear",
        ).to(torch.float32)

        w_mean = self.mean_features_weight
        w_inpaint = 1 - w_mean

        if w_mean == 0:
            return texture

        texture_mean = aggregate_views_uv_texture_mean(
            feature_maps,
            uv_res,
            projections,
            interpolation_mode="bilinear",
        ).to(torch.float32)

        return w_mean * texture_mean + w_inpaint * texture

    def calculate_layer_resolutions(self):
        layers = [AttnLayerId.parse(path) for path in self.module_paths]
        layer_resolutions = list(
            set([layer.resolution(self.pipe.unet) for layer in layers])
        )
        layer_resolutions = sorted(layer_resolutions)

        layer_resolution_indices = {
            layer.module_path(): layer_resolutions.index(
                layer.resolution(self.pipe.unet)
            )
            for layer in layers
        }

        return layer_resolutions, layer_resolution_indices

    def precompute_seq(self, anim: AnimSequence, resolutions, uv_resolutions):
        projections, fragments = precompute_rasterization(
            anim.cams,
            anim.meshes,
            anim.verts_uvs,
            anim.faces_uvs,
            resolutions,
            uv_resolutions,
        )

        depth_maps = render_depth_map(anim.meshes, anim.cams)

        return GrCamSeq(anim, projections, fragments, depth_maps)


def sample_keyframe_indices(
    n_frames: int, num_keyframes: int, generator: torch.Generator = None, device="cuda"
):
    if num_keyframes > n_frames:
        raise ValueError(
            f"Number of keyframes ({num_keyframes}) is greater than number of frames ({n_frames})."
        )

    randperm = torch.randperm(n_frames, generator=generator, device=device)
    return randperm[:num_keyframes]


@dataclass
class GenerativeRenderingOutput:
    images: List[Image.Image]
    extr_images: Optional[List[Image.Image]] = None


class RgbExtractionPipeline(BaseControlNetPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        anim: AnimSequence,
        src_frames: List[Image.Image],
        generator=None,
    ):
        n_frames = len(anim)

        cond_embs, uncond_embs = self.encode_prompt([prompt] * n_frames)

        latents = self.prepare_latents(n_frames, generator)

        self.scheduler.set_timesteps(10)

        depth_maps = anim.render_depth_maps()

        src_latents = self.encode_images(src_frames)
        src_noise = torch.randn_like(src_latents)

        decoder_paths = [
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

        gr = GenerativeRenderingLogic(self, module_paths=decoder_paths)
        gr.set_attn_processor()

        for t in tqdm(self.scheduler.timesteps):
            src_noisy = self.scheduler.add_noise(src_latents, src_noise, t)

            feats = gr.model_forward_extraction(
                src_noisy, cond_embs, uncond_embs, depth_maps, t
            )

            noise_pred = gr.model_forward_injection(
                latents,
                cond_embs,
                uncond_embs,
                depth_maps,
                t,
                {},
                {},
                feats.cond_post_attn,
                feats.uncond_post_attn,
            )

            latents = self.scheduler.step(
                noise_pred, t, latents, generator=generator
            ).prev_sample

        return self.decode_latents(latents, generator=generator)


class GenerativeRenderingPipeline(BaseControlNetPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        anim: AnimSequence,
        conf: GenerativeRenderingConfig,
        src_anim: Optional[AnimSequence] = None,
        texture: Optional[Tensor] = None,
        start_noise_level: Optional[float] = 0,
        kf_generator: Optional[Generator] = None,
        generator: Optional[Generator] = None,
        logger=NULL_LOGGER,
    ):
        n_frames = len(anim.cams)

        noise_texture = UVNoiseInitializer(noise_texture_res=120)

        # setup GR logic
        gr = GenerativeRenderingLogic(
            self,
            module_paths=conf.module_paths,
            guidance_scale=conf.guidance_scale,
            controlnet_conditioning_scale=conf.controlnet_conditioning_scale,
            mean_features_weight=conf.mean_features_weight,
            do_pre_attn_injection=conf.do_pre_attn_injection,
            do_post_attn_injection=conf.do_post_attn_injection,
        )
        gr.set_attn_processor()

        use_texture = texture is not None

        # configure scheduler
        if use_texture:
            timesteps = self.get_partial_timesteps(
                conf.num_inference_steps, start_noise_level
            )
        else:
            timesteps = self.get_partial_timesteps(conf.num_inference_steps, 0)

        setup_greenlists(logger, timesteps, n_frames, n_save_frames=10, n_save_times=10)

        layer_resolutions, layer_resolution_indices = gr.calculate_layer_resolutions()
        uv_resolutions = [4 * res for res in layer_resolutions]

        precomputed = gr.precompute_seq(anim, layer_resolutions, uv_resolutions)

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt] * n_frames)

        # initialize latents
        # TODO support taking in the float and the texture and starting denoising from there
        noise_texture.sample_background(generator)
        noise_texture.sample_noise_texture(generator)

        if use_texture:
            anim_renders = render_texture(
                anim.meshes, anim.cams, texture, anim.verts_uvs, anim.faces_uvs
            )

            anim_encoded = self.encode_images(anim_renders)
            anim_noise = noise_texture.initial_noise(
                anim.meshes, anim.cams, anim.verts_uvs, anim.faces_uvs
            )

            for i in range(len(anim_renders)):
                logger.write("anim_render", anim_renders[i], frame_i=i)
                logger.write("anim_encoded", anim_encoded[i], frame_i=i)

            anim_latents = self.scheduler.add_noise(
                anim_encoded, anim_noise, timesteps[0]
            )

        else:
            anim_latents = noise_texture.initial_noise(
                anim.meshes,
                anim.cams,
                anim.verts_uvs,
                anim.faces_uvs,
                dtype=self.dtype,
                device=self.device,
                generator=generator,
            )

        for i in range(len(anim_latents)):
            logger.write("anim_latent", anim_latents[i], frame_i=i)

        # use kfs/or src seq
        use_keyframes = src_anim is None

        # initialize latents for extraction frames
        if not use_keyframes:
            if use_texture:
                extr_renders = render_texture(
                    src_anim.meshes,
                    src_anim.cams,
                    texture,
                    src_anim.verts_uvs,
                    src_anim.faces_uvs,
                )

                extr_encoded = self.encode_images(extr_renders)

                for i in range(len(extr_renders)):
                    logger.write("extr_render", extr_renders[i], extr_frame_i=i)
                    logger.write("extr_encoded", extr_encoded[i], extr_frame_i=i)

                extr_noise = noise_texture.initial_noise(
                    src_anim.meshes,
                    src_anim.cams,
                    src_anim.verts_uvs,
                    src_anim.faces_uvs,
                )
                extr_latents = self.scheduler.add_noise(
                    extr_encoded, extr_noise, timesteps[0]
                )

            else:
                extr_latents = noise_texture.initial_noise(
                    src_anim.meshes,
                    src_anim.cams,
                    src_anim.verts_uvs,
                    src_anim.faces_uvs,
                    dtype=self.dtype,
                    device=self.device,
                    generator=generator,
                )

            for i in range(len(extr_latents)):
                logger.write("extr_latent", extr_latents[i], extr_frame_i=i)

            precomputed_extraction = gr.precompute_seq(
                src_anim, layer_resolutions, uv_resolutions
            )

            extr_prompts = [prompt] * len(src_anim.cams)
            extr_cond_embs, extr_uncond_embs = self.encode_prompt(extr_prompts)

            extr_depth_maps = precomputed_extraction.depth_maps
            extr_projections = list(precomputed_extraction.projections.values())
            extr_fragments = list(precomputed_extraction.fragments.values())

        # chunk indices to use i regeneration
        chunks_indices = torch.split(torch.arange(0, n_frames), conf.chunk_size)

        # denoising loop
        for t in tqdm(timesteps):
            if use_keyframes:
                kf_indices = sample_keyframe_indices(
                    n_frames, conf.num_keyframes, kf_generator
                )

                # get extr_latents/embs from anim_latents/embs
                extr_latents = anim_latents[kf_indices]
                extr_cond_embs = cond_embeddings[kf_indices]
                extr_uncond_embs = uncond_embeddings[kf_indices]
                extr_depth_maps = [
                    precomputed.depth_maps[i] for i in kf_indices.tolist()
                ]
                extr_projections = [
                    precomputed.projections[i] for i in kf_indices.tolist()
                ]

            feats = gr.model_forward_extraction(
                extr_latents,
                extr_cond_embs,
                extr_uncond_embs,
                extr_depth_maps,
                t,
            )

            def aggr_feats(layer, features):
                res_idx = layer_resolution_indices[layer]
                uv_res = uv_resolutions[res_idx]
                projections = [projs[res_idx] for projs in extr_projections]
                return gr.aggregate_features(features, uv_res, projections)

            textures_uncond = dict_map(feats.uncond_post_attn, aggr_feats)
            textures_cond = dict_map(feats.cond_post_attn, aggr_feats)

            # write extracted features
            write_feature_dict(logger, "kvs_cond", feats.cond_kv, t)
            write_feature_frame_dict(
                logger,
                "feats_cond",
                feats.cond_post_attn,
                t,
                torch.arange(0, len(extr_latents)),
                frame_key="extr_frame_i",
            )

            # write feature texture
            write_feature_dict(logger, "feat_tex_cond", textures_cond, t)

            progress = self.denoising_progress(t)

            feature_alpha = linear_map(
                progress, from_min=0, from_max=1, to_min=1, to_max=0.3
            )
            print(feature_alpha)

            # denoise anim latents
            noise_preds = []
            for chunk_frame_indices in chunks_indices:
                # render chunk feats
                chunk_frags = [
                    precomputed.fragments[i] for i in chunk_frame_indices.tolist()
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

                write_feature_frame_dict(
                    logger, "renders_cond", renders_cond, t, chunk_frame_indices
                )

                # model forward with pre-and post-attn injection
                noise_pred = gr.model_forward_injection(
                    anim_latents[chunk_frame_indices],
                    cond_embeddings[chunk_frame_indices],
                    uncond_embeddings[chunk_frame_indices],
                    [precomputed.depth_maps[i] for i in chunk_frame_indices.tolist()],
                    t,
                    feats.cond_kv,
                    feats.uncond_kv,
                    renders_cond,
                    renders_uncond,
                    progress=feature_alpha,
                )

                noise_preds.append(noise_pred)

            noise_preds = torch.cat(noise_preds, dim=0)
            anim_latents = self.scheduler.step(
                noise_preds, t, anim_latents, generator=generator
            ).prev_sample

            # denoise extraction latents
            if not use_keyframes:

                def render_extr_feats(layer, uv_map):
                    extr_meshes = src_anim.meshes
                    res_idx = layer_resolution_indices[layer]
                    frags = [f[res_idx] for f in extr_fragments]
                    texture = make_repeated_uv_texture(
                        uv_map,
                        anim.faces_uvs,
                        anim.verts_uvs,
                        sampling_mode="nearest",
                        N=1,
                    )
                    shader = TextureShader()
                    return shade_meshes(shader, texture, extr_meshes, frags)

                renders_uncond = dict_map(textures_uncond, render_extr_feats)
                renders_cond = dict_map(textures_cond, render_extr_feats)

                # model forward with pre-and post-attn injection
                noise_pred = gr.model_forward_injection(
                    extr_latents,
                    extr_cond_embs,
                    extr_uncond_embs,
                    extr_depth_maps,
                    t,
                    feats.cond_kv,
                    feats.uncond_kv,
                    renders_cond,
                    renders_uncond,
                    progress=feature_alpha,
                )

                extr_latents = self.scheduler.step(
                    noise_pred, t, extr_latents, generator=generator
                ).prev_sample

        ims = self.decode_latents(
            anim_latents, chunk_size=conf.chunk_size, generator=generator
        )

        if not use_keyframes:
            extr_ims = self.decode_latents(
                extr_latents, chunk_size=conf.chunk_size, generator=generator
            )
        else:
            extr_ims = None

        return GenerativeRenderingOutput(images=ims, extr_images=extr_ims)
