from typing import Dict, List

import torch
from attr import dataclass
from diffusers.schedulers.scheduling_utils import SchedulerMixin
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
    TexelProjection,
    aggregate_views_uv_texture,
    aggregate_views_uv_texture_mean,
)
from text3d2video.noise_initialization import NoiseInitializer
from text3d2video.pipelines.controlnet_pipeline import BaseControlNetPipeline
from text3d2video.rendering import (
    TextureShader,
    make_repeated_uv_texture,
    precompute_rasterization,
    render_depth_map,
    shade_meshes,
)
from text3d2video.sd_feature_extraction import AttnLayerId
from text3d2video.util import dict_map
from text3d2video.utilities.diffusion_data import (
    DiffusionDataLogger,
)


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


class GenerativeRenderingLogger(DiffusionDataLogger):
    def __init__(self, h5_file_path, enabled=True, n_save_frames=5, n_save_levels=8):
        self.n_frames = n_save_frames
        self.n_times = n_save_levels
        super().__init__(
            h5_file_path,
            enabled,
            [],
            [],
            [],
        )

    def setup_logger(
        self, scheduler: SchedulerMixin, n_frames: int, modules: list[str]
    ):
        self.path_greenlist = modules
        self.calc_evenly_spaced_frame_indices(n_frames, self.n_frames)
        self.calc_evenly_spaced_noise_noise_levels(scheduler, self.n_times)


class GenerativeRenderingPipeline(BaseControlNetPipeline):
    attn_processor: ExtractionInjectionAttn
    conf: GenerativeRenderingConfig
    noise_initializer: NoiseInitializer
    logger: GenerativeRenderingLogger

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
        if self.conf.num_keyframes > n_frames:
            raise ValueError("Number of keyframes is greater than number of frames")

        randperm = torch.randperm(n_frames, generator=generator, device=self.device)
        return randperm[: self.conf.num_keyframes]

    def model_forward(
        self,
        latents: Float[Tensor, "b c h w"],
        embeddings: Float[Tensor, "b t d"],
        t: int,
        depth_maps: List[Image.Image],
    ) -> Tensor:
        # ControlNet Pass
        processed_ctrl_images = self.preprocess_controlnet_images(depth_maps)
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latents,
            t,
            encoder_hidden_states=embeddings,
            controlnet_cond=processed_ctrl_images,
            conditioning_scale=self.conf.controlnet_conditioning_scale,
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
    ) -> GrExtractedFeatures:
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

        # classifier-free guidance
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred_guided = noise_pred_uncond + self.conf.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        return noise_pred_guided

    def aggregate_features(
        self,
        feature_maps: Float[Tensor, "b c h w"],
        uv_res: int,
        projections: List[TexelProjection],
    ) -> Float[Tensor, "h w c"]:
        texel_xys = [proj.xys for proj in projections]
        texel_uvs = [proj.uvs for proj in projections]

        texture = aggregate_views_uv_texture(
            feature_maps,
            uv_res,
            texel_xys,
            texel_uvs,
            interpolation_mode="bilinear",
        ).to(torch.float32)

        w_mean = self.conf.mean_features_weight
        w_inpaint = 1 - w_mean

        if w_mean == 0:
            return texture

        texture_mean = aggregate_views_uv_texture_mean(
            feature_maps,
            uv_res,
            texel_xys,
            texel_uvs,
            interpolation_mode="bilinear",
        ).to(torch.float32)

        w_mean = self.conf.mean_features_weight
        w_inpaint = 1 - w_mean
        return w_mean * texture_mean + w_inpaint * texture

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
        logger=None,
    ):
        n_frames = len(meshes)

        # setup configs
        self.conf = generative_rendering_config
        self.noise_initializer = noise_initializer

        # configure scheduler
        self.scheduler.set_timesteps(self.conf.num_inference_steps)

        # precompute rasterization and texel projection for various resolutions (for different layers)
        layers = [AttnLayerId.parse(path) for path in self.conf.module_paths]
        layer_resolutions = list(set([layer.resolution(self.unet) for layer in layers]))
        uv_resolutions = [screen * 4 for screen in layer_resolutions]

        layer_resolution_indices = {
            layer.module_path(): layer_resolutions.index(layer.resolution(self.unet))
            for layer in layers
        }

        projections, fragments = precompute_rasterization(
            cameras, meshes, verts_uvs, faces_uvs, layer_resolutions, uv_resolutions
        )

        # setup logger
        if logger is not None:
            self.logger = logger
            self.logger.setup_logger(
                self.scheduler, n_frames, generative_rendering_config.module_paths
            )

        # set up attn processor
        self.attn_processor = ExtractionInjectionAttn(
            self.unet,
            do_spatial_post_attn_extraction=self.conf.do_post_attn_injection,
            do_kv_extraction=self.conf.do_pre_attn_injection,
            also_attend_to_self=self.conf.attend_to_self_kv,
            feature_blend_alpha=self.conf.feature_blend_alpha,
            kv_extraction_paths=self.conf.module_paths,
            spatial_post_attn_extraction_paths=self.conf.module_paths,
            unet_chunk_size=2,
        )
        self.unet.set_attn_processor(self.attn_processor)

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt] * n_frames)

        # initial latent noise
        latents = self.prepare_latents(meshes, cameras, verts_uvs, faces_uvs, generator)

        # chunk indices to use in inference loop
        chunks_indices = torch.split(torch.arange(0, n_frames), self.conf.chunk_size)

        # render depth maps for frames
        depth_maps = render_depth_map(meshes, cameras, 512)

        # denoising loop
        for t in tqdm(self.scheduler.timesteps):
            self.attn_processor.cur_timestep = t

            # Feature Extraction on keyframes
            kf_indices = self.sample_keyframe_indices(n_frames, generator)

            kf_features = self.model_forward_extraction(
                latents[kf_indices],
                cond_embeddings[kf_indices],
                uncond_embeddings[kf_indices],
                [depth_maps[i] for i in kf_indices.tolist()],
                t,
            )

            def aggr_kf_features(layer, features):
                res_idx = layer_resolution_indices[layer]
                uv_res = uv_resolutions[res_idx]
                kf_projections = [projections[i][res_idx] for i in kf_indices.tolist()]
                return self.aggregate_features(features, uv_res, kf_projections)

            textures_uncond = dict_map(kf_features.uncond_post_attn, aggr_kf_features)
            textures_cond = dict_map(kf_features.cond_post_attn, aggr_kf_features)

            # denoising in chunks
            noise_preds = []
            for chunk_frame_indices in chunks_indices:
                # render chunk post-attn features
                def render_chunk(layer, uv_map):
                    chunk_meshes = meshes[chunk_frame_indices]
                    res_idx = layer_resolution_indices[layer]
                    chunk_frags = [
                        fragments[i][res_idx] for i in chunk_frame_indices.tolist()
                    ]

                    texture = make_repeated_uv_texture(
                        uv_map, faces_uvs, verts_uvs, sampling_mode="nearest"
                    )
                    shader = TextureShader()

                    return shade_meshes(shader, texture, chunk_meshes, chunk_frags)

                renders_uncond = dict_map(textures_uncond, render_chunk)
                renders_cond = dict_map(textures_cond, render_chunk)

                # Diffusion step with pre-and post-attn injection
                noise_pred = self.model_forward_injection(
                    latents[chunk_frame_indices],
                    cond_embeddings[chunk_frame_indices],
                    uncond_embeddings[chunk_frame_indices],
                    [depth_maps[i] for i in chunk_frame_indices.tolist()],
                    t,
                    kf_features.cond_kv,
                    kf_features.uncond_kv,
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
