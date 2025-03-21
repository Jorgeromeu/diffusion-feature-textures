from typing import Dict, List

import torch
from attr import dataclass
from jaxtyping import Float
from PIL import Image
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
)
from pytorch3d.structures import Meshes
from regex import B
from torch import Tensor
from tqdm import tqdm

from text3d2video.attn_processors.extraction_injection_attn import (
    ExtractionInjectionAttn,
)
from text3d2video.noise_initialization import NoiseInitializer, UVNoiseInitializer
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingPipeline,
)
from text3d2video.rendering import (
    TextureShader,
    make_mesh_renderer,
    make_repeated_uv_texture,
    precompute_rast_fragments,
    render_depth_map,
)
from text3d2video.sd_feature_extraction import AttnLayerId
from text3d2video.util import dict_filter, dict_map
from text3d2video.utilities.logging import GrLogger, H5Logger


@dataclass
class FeatureInjectionConfig:
    do_pre_attn_injection: bool
    do_post_attn_injection: bool
    feature_blend_alpha: float
    attend_to_self_kv: bool
    chunk_size: int
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    time_threshold: float
    layer_threshold: float
    module_paths: list[str]


class ReposableDiffusion2Pipeline(GenerativeRenderingPipeline):
    conf: FeatureInjectionConfig
    attn_processor: ExtractionInjectionAttn

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

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        tgt_meshes: Meshes,
        tgt_cams: FoVPerspectiveCameras,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        reposable_diffusion_config: FeatureInjectionConfig,
        extracted_feats: H5Logger,
        generator=None,
        logger: GrLogger = None,
    ):
        # setup configs for use throughout pipeline
        self.conf = reposable_diffusion_config

        # setup attn processor
        self.attn_processor = ExtractionInjectionAttn(
            self.unet,
            do_spatial_post_attn_extraction=self.conf.do_post_attn_injection,
            do_kv_extraction=self.conf.do_pre_attn_injection,
            also_attend_to_self=self.conf.attend_to_self_kv,
            feature_blend_alpha=self.conf.feature_blend_alpha,
            kv_extraction_paths=self.conf.module_paths,
            spatial_post_attn_extraction_paths=self.conf.module_paths,
        )
        self.unet.set_attn_processor(self.attn_processor)

        # configure scheduler
        self.scheduler.set_timesteps(self.conf.num_inference_steps)

        n_frames = len(tgt_meshes)
        frame_indices = torch.arange(n_frames)

        # render all depth maps
        depth_maps = render_depth_map(tgt_meshes, tgt_cams, 512)

        layers = [AttnLayerId.parse(path) for path in self.conf.module_paths]
        screen_resolutions = list(
            set([layer.resolution(self.unet) for layer in layers])
        )

        layer_resolution_indices = {
            layer.module_path(): screen_resolutions.index(layer.resolution(self.unet))
            for layer in layers
        }

        # filter layers to perform injection on
        max_layer_index = int(len(layers) * self.conf.layer_threshold)
        injection_layers = self.conf.module_paths[0:max_layer_index]

        fragments = precompute_rast_fragments(tgt_cams, tgt_meshes, screen_resolutions)

        # setup logger
        if logger is not None:
            self.logger = logger
            self.logger.setup_greenlists(self.scheduler.timesteps.tolist(), n_frames)
        else:
            self.logger = GrLogger.create_disabled()

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt] * n_frames)

        # compute maximum time at which we perform injection
        denoising_times = self.scheduler.timesteps
        max_t_idx = int(len(denoising_times) * self.conf.time_threshold)
        max_t_idx = min(max_t_idx, len(denoising_times) - 1)
        max_t = denoising_times[max_t_idx]

        uv_noise = extracted_feats.read("uv_noise", return_pt=True).to(cond_embeddings)
        bg_noise = extracted_feats.read("bg_noise", return_pt=True).to(cond_embeddings)

        noise_initializer = UVNoiseInitializer.init_from_textures(uv_noise, bg_noise)
        latents = noise_initializer.initial_noise(
            tgt_meshes,
            tgt_cams,
            verts_uvs,
            faces_uvs,
        )

        # denoising loop
        for t_i, t in enumerate(tqdm(self.scheduler.timesteps)):
            self.attn_processor.set_cur_timestep(t)

            def read_features(t, name):
                return {
                    layer: extracted_feats.read(
                        name, layer=layer, t=int(t), return_pt=True
                    ).to(device=self.device, dtype=self.dtype)
                    for layer in self.conf.module_paths
                }

            kvs_cond = read_features(t, "kvs_cond")
            kvs_uncond = read_features(t, "kvs_uncond")
            textures_cond = read_features(t, "tex_cond")
            textures_uncond = read_features(t, "tex_uncond")

            if not self.conf.do_post_attn_injection:
                textures_cond = {}
                textures_uncond = {}

            if not self.conf.do_pre_attn_injection:
                kvs_cond = {}
                kvs_uncond = {}

            def filter(layer, _):
                return layer in injection_layers

            textures_cond = dict_filter(textures_cond, filter)
            textures_uncond = dict_filter(textures_uncond, filter)

            # Denoise target frames in chunks with injection
            noise_preds = []
            for chunk_frame_indices in torch.split(frame_indices, self.conf.chunk_size):

                def render_chunk_frames(layer, texture):
                    shader = TextureShader()
                    tex = make_repeated_uv_texture(
                        texture,
                        faces_uvs,
                        verts_uvs,
                        sampling_mode="nearest",
                    )

                    res_idx = layer_resolution_indices[layer]

                    renders = []
                    for frame_i in chunk_frame_indices.tolist():
                        mesh = tgt_meshes[frame_i]
                        frags = fragments[res_idx][frame_i]
                        mesh.textures = tex
                        render = shader(frags, mesh)[0]

                        renders.append(render)
                    renders = torch.stack(renders)
                    return renders

                chunk_rendered_cond = dict_map(textures_cond, render_chunk_frames)
                chunk_rendered_uncond = dict_map(textures_uncond, render_chunk_frames)

                if t < max_t:
                    chunk_rendered_cond = {}
                    chunk_rendered_uncond = {}

                self.logger.write_rendered_features(
                    "rendered_cond",
                    chunk_rendered_cond,
                    t=t,
                    frame_indices=chunk_frame_indices,
                )

                chunk_noise = self.model_forward_injection(
                    latents[chunk_frame_indices],
                    cond_embeddings[chunk_frame_indices],
                    uncond_embeddings[chunk_frame_indices],
                    [depth_maps[i] for i in chunk_frame_indices.tolist()],
                    t,
                    kvs_cond,
                    kvs_uncond,
                    chunk_rendered_cond,
                    chunk_rendered_uncond,
                )

                noise_preds.append(chunk_noise)

            # Denoise all
            noise_preds = torch.cat(noise_preds)
            latents = self.scheduler.step(
                noise_preds, t, latents, generator=generator
            ).prev_sample

        # decode latents in chunks
        decoded_imgs = []
        for chunk_frame_indices in torch.split(frame_indices, self.conf.chunk_size):
            chunk_latents = latents[chunk_frame_indices]
            chunk_images = self.decode_latents(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        return decoded_imgs
