from typing import Dict, List

import torch
from attr import dataclass
from jaxtyping import Float
from PIL import Image
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
)
from pytorch3d.structures import Meshes
from torch import Tensor
from tqdm import tqdm

from text3d2video.attn_processors.extraction_injection_attn import (
    ExtractionInjectionAttn,
)
from text3d2video.backprojection import (
    aggregate_views_vert_texture,
    project_visible_texels_to_camera,
    project_visible_verts_to_camera,
)
from text3d2video.noise_initialization import NoiseInitializer
from text3d2video.pipelines.controlnet_pipeline import BaseControlNetPipeline
from text3d2video.rendering import (
    make_mesh_rasterizer,
    make_mesh_renderer,
    make_repeated_vert_texture,
    render_depth_map,
)
from text3d2video.util import map_dict


# pylint: disable=too-many-instance-attributes
@dataclass
class GenerativeRenderingConfig:
    do_pre_attn_injection: bool
    do_post_attn_injection: bool
    feature_blend_alpha: float
    attend_to_self_kv: bool
    mean_features_weight: float
    chunk_size: int
    num_keyframes: int
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    module_paths: list[str]


@dataclass
class ExtractedFeatures:
    cond_kv_features: Dict[str, Float[torch.Tensor, "b t d"]]
    uncond_kv_features: Dict[str, Float[torch.Tensor, "b t d"]]
    cond_post_attn_features: Dict[str, Float[torch.Tensor, "b f t d"]]
    uncond_post_attn_features: Dict[str, Float[torch.Tensor, "b f t d"]]


class GenerativeRenderingPipeline(BaseControlNetPipeline):
    attn_processor: ExtractionInjectionAttn
    rd_config: GenerativeRenderingConfig
    noise_initializer: NoiseInitializer

    def prepare_latents(
        self,
        meshes: Meshes,
        cameras: FoVPerspectiveCameras,
        verts_uvs,
        faces_uvs,
        generator=None,
    ):
        return self.noise_initializer.initial_noise(
            cameras=cameras,
            meshes=meshes,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
            device=self.device,
            dtype=self.dtype,
            n_frames=len(meshes),
            generator=generator,
        )

    def sample_keyframe_indices(
        self, n_frames: int, generator: torch.Generator = None
    ) -> torch.Tensor:
        if self.rd_config.num_keyframes > n_frames:
            raise ValueError("Number of keyframes is greater than number of frames")

        randperm = torch.randperm(n_frames, generator=generator, device=self.device)
        return randperm[: self.rd_config.num_keyframes]

    def model_forward(
        self,
        latents: Float[Tensor, "b c h w"],
        embeddings: Float[Tensor, "b t d"],
        t: int,
        depth_maps: List[Image.Image],
    ):
        # ControlNet Pass
        processed_ctrl_images = self.preprocess_controlnet_images(depth_maps)
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latents,
            t,
            encoder_hidden_states=embeddings,
            controlnet_cond=processed_ctrl_images,
            conditioning_scale=self.rd_config.controlnet_conditioning_scale,
            guess_mode=False,
            return_dict=False,
        )

        # UNet Pass
        noise_pred = self.unet(
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
    ):
        # do cond and uncond passes
        latents_duplicated = torch.cat([latents] * 2)
        both_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        depth_maps_duplicated = depth_maps * 2

        # model pass, to extract features
        self.attn_processor.set_extraction_mode()
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

        return ExtractedFeatures(
            cond_kv_features=extracted_kv_cond,
            uncond_kv_features=extracted_kv_uncond,
            cond_post_attn_features=extracted_post_attn_cond,
            uncond_post_attn_features=extracted_post_attn_uncond,
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
    ):
        latents_duplicated = torch.cat([latents] * 2)
        embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        depth_maps_duplicated = depth_maps * 2

        injected_kvs = {}
        injected_post_attn = {}
        for layer in cond_kv_features.keys():
            layer_kvs = torch.stack(
                [uncond_kv_features[layer], uncond_kv_features[layer]]
            )
            injected_kvs[layer] = layer_kvs

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

        # classifier-free guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred_guided = noise_pred_uncond + self.rd_config.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        return noise_pred_guided

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        meshes: Meshes,
        cameras: FoVPerspectiveCameras,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        generative_rendering_config: GenerativeRenderingConfig,
        noise_initializer: NoiseInitializer,
        generator=None,
    ):
        n_frames = len(meshes)

        # setup configs for use throughout pipeline
        self.rd_config = generative_rendering_config
        self.noise_initializer = noise_initializer

        # set up attn processor
        self.attn_processor = ExtractionInjectionAttn(
            self.unet,
            do_spatial_post_attn_extraction=self.rd_config.do_post_attn_injection,
            do_kv_extraction=self.rd_config.do_pre_attn_injection,
            also_attend_to_self=self.rd_config.attend_to_self_kv,
            feature_blend_alpha=self.rd_config.feature_blend_alpha,
            kv_extraction_paths=self.rd_config.module_paths,
            spatial_post_attn_extraction_paths=self.rd_config.module_paths,
            unet_chunk_size=2,
        )
        self.unet.set_attn_processor(self.attn_processor)

        # configure scheduler
        self.scheduler.set_timesteps(self.rd_config.num_inference_steps)

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt] * n_frames)

        # initial latent noise
        latents = self.prepare_latents(meshes, cameras, verts_uvs, faces_uvs, generator)

        # chunk indices to use in inference loop
        chunks_indices = torch.split(
            torch.arange(0, n_frames), self.rd_config.chunk_size
        )

        # render depth maps for frames
        depth_maps = render_depth_map(meshes, cameras, 512)

        # precompute texture rasterization for efficient projection
        uv_res = 350
        texel_xys = []
        texel_uvs = []
        vert_xys = []
        vert_indices = []
        for cam, mesh in zip(cameras, meshes):
            xys, indices = project_visible_verts_to_camera(mesh, cam)
            vert_xys.append(xys)
            vert_indices.append(indices)

            xys, uvs = project_visible_texels_to_camera(
                mesh, cam, verts_uvs, faces_uvs, uv_res, raster_res=2000
            )
            texel_xys.append(xys)
            texel_uvs.append(uvs)

        # denoising loop
        for t in tqdm(self.scheduler.timesteps):
            self.attn_processor.cur_timestep = t

            # sample keyframes
            kf_indices = self.sample_keyframe_indices(n_frames, generator)

            # Feature Extraction on keyframes
            extracted_features = self.model_forward_extraction(
                latents[kf_indices],
                cond_embeddings[kf_indices],
                uncond_embeddings[kf_indices],
                [depth_maps[i] for i in kf_indices.tolist()],
                t,
            )

            # Aggregate KF features to UV space
            layer_resolutions = map_dict(
                extracted_features.cond_post_attn_features, lambda _, x: x.shape[-1]
            )

            kf_texel_xys = [texel_xys[i] for i in kf_indices.tolist()]
            kf_texel_uvs = [texel_uvs[i] for i in kf_indices.tolist()]
            kf_vert_xys = [vert_xys[i] for i in kf_indices.tolist()]
            kf_vert_indices = [vert_indices[i] for i in kf_indices.tolist()]

            def aggregate(layer, features):
                texture = aggregate_views_vert_texture(
                    features,
                    meshes.num_verts_per_mesh()[0],
                    kf_vert_xys,
                    kf_vert_indices,
                    mode="nearest",
                    aggregation_type="first",
                ).to(torch.float32)

                return texture

            aggregated_uncond = map_dict(
                extracted_features.uncond_post_attn_features, aggregate
            )
            aggregated_cond = map_dict(
                extracted_features.cond_post_attn_features, aggregate
            )

            # denoising in chunks
            noise_preds = []
            for chunk_frame_indices in chunks_indices:
                chunk_cams = cameras[chunk_frame_indices]
                chunk_meshes = meshes[chunk_frame_indices]

                # Render
                def render(layer, texture):
                    tex = make_repeated_vert_texture(texture, len(chunk_cams))
                    tex.sampling_mode = "nearest"

                    renderer = make_mesh_renderer(
                        cameras=chunk_cams, resolution=layer_resolutions[layer]
                    )

                    render_meshes = chunk_meshes.clone()
                    render_meshes.textures = tex
                    return renderer(render_meshes)

                renders_uncond = map_dict(aggregated_uncond, render)
                renders_cond = map_dict(aggregated_cond, render)

                # Diffusion step #2 with pre and post attn feature injection
                # get chunk inputs
                chunk_latents = latents[chunk_frame_indices]
                chunk_cond_embeddings = cond_embeddings[chunk_frame_indices]
                chunk_uncond_embeddings = uncond_embeddings[chunk_frame_indices]
                chunk_depth_maps = [depth_maps[i] for i in chunk_frame_indices.tolist()]

                noise_pred = self.model_forward_injection(
                    chunk_latents,
                    chunk_cond_embeddings,
                    chunk_uncond_embeddings,
                    chunk_depth_maps,
                    t,
                    extracted_features.cond_kv_features,
                    extracted_features.uncond_kv_features,
                    renders_cond,
                    renders_uncond,
                )

                noise_preds.append(noise_pred)

            # concatenate predictions
            noise_preds = torch.cat(noise_preds, dim=0)

            # update latents
            latents = self.scheduler.step(
                noise_preds, t, latents, generator=generator
            ).prev_sample

        # decode latents in chunks
        decoded_imgs = []
        for chunk_frame_indices in chunks_indices:
            chunk_latents = latents[chunk_frame_indices]
            chunk_images = self.decode_latents(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        return decoded_imgs
