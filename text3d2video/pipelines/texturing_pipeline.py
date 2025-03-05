from typing import Dict, List, Tuple

import torch
from attr import dataclass
from einops import rearrange
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
    aggregate_spatial_features_dict,
    project_visible_verts_to_camera,
    rasterize_and_render_vert_features_dict,
)
from text3d2video.noise_initialization import NoiseInitializer
from text3d2video.pipelines.controlnet_pipeline import BaseControlNetPipeline
from text3d2video.rendering import render_depth_map


# pylint: disable=too-many-instance-attributes
@dataclass
class TexturingConfig:
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


class TexturingPipeline(BaseControlNetPipeline):
    attn_processor: ExtractionInjectionAttn
    conf: TexturingConfig

    def model_forward(
        self,
        latents: Float[Tensor, "b f c h w"],
        text_embeddings: Float[Tensor, "b f t d"],
        t: int,
        depth_maps: List[Image.Image],
    ) -> Float[Tensor, "b f c h w"]:
        """
        Forward pass of the controlnet and unet
        """

        # batch across time dimension
        chunk_size = latents.shape[0]
        batched_latents = rearrange(latents, "b f c h w -> (b f) c h w")
        batched_embeddings = rearrange(text_embeddings, "b f t d -> (b f) t d")

        # controlnet step
        processed_ctrl_images = self.preprocess_controlnet_images(depth_maps)
        processed_ctrl_images = torch.cat([processed_ctrl_images] * 2)

        down_block_res_samples, mid_block_res_sample = self.controlnet(
            batched_latents,
            t,
            encoder_hidden_states=batched_embeddings,
            controlnet_cond=processed_ctrl_images,
            conditioning_scale=self.conf.controlnet_conditioning_scale,
            guess_mode=False,
            return_dict=False,
        )

        # unet, with controlnet residuals
        noise_pred = self.unet(
            batched_latents,
            t,
            mid_block_additional_residual=mid_block_res_sample,
            down_block_additional_residuals=down_block_res_samples,
            encoder_hidden_states=batched_embeddings,
        ).sample

        # unbatch
        noise_pred = rearrange(noise_pred, "(b f) c h w -> b f c h w", b=chunk_size)

        return noise_pred

    def model_fwd_feature_extraction(
        self,
        latents: Float[Tensor, "b f c h w"],
        text_embeddings: Float[Tensor, "b f t d"],
        depth_maps: List[Image.Image],
        t: int,
    ) -> Tuple[
        Dict[str, Float[torch.Tensor, "b t c"]],
        Dict[str, Float[torch.Tensor, "b f t c"]],
    ]:
        # forward pass
        self.attn_processor.set_extraction_mode()
        self.model_forward(latents, text_embeddings, t, depth_maps=depth_maps)

        # get saved features
        pre_attn_features = self.attn_processor.kv_features
        post_attn_features = self.attn_processor.spatial_post_attn_features

        self.attn_processor.clear_features()

        return pre_attn_features, post_attn_features

    def model_fwd_feature_injection(
        self,
        latents: Float[Tensor, "b f c h w"],
        text_embeddings: Float[Tensor, "b f t d"],
        depth_maps: List[Image.Image],
        t: int,
        pre_attn_features: Dict[str, Float[Tensor, "b f t d"]],
        feature_images: Dict[str, Float[Tensor, "b f d h w"]],
        frame_indices: Tensor,
    ):
        # set injection mode and pass features
        self.attn_processor.set_injection_mode(
            pre_attn_features=pre_attn_features,
            post_attn_features=feature_images,
        )

        # pass frame indices to attn processor
        self.attn_processor.set_chunk_frame_indices(frame_indices)
        noise_pred = self.model_forward(
            latents, text_embeddings, t, depth_maps=depth_maps
        )

        return noise_pred

    def sample_keyframe_indices(
        self, n_frames: int, generator: torch.Generator = None
    ) -> torch.Tensor:
        if self.conf.num_keyframes > n_frames:
            raise ValueError("Number of keyframes is greater than number of frames")

        randperm = torch.randperm(n_frames, generator=generator, device=self.device)
        return randperm[: self.conf.num_keyframes]

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        meshes: Meshes,
        cameras: FoVPerspectiveCameras,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        conf: TexturingConfig,
        generator=None,
    ):
        # setup configs for use throughout pipeline
        self.conf = conf

        # set up attn processor
        self.attn_processor = ExtractionInjectionAttn(
            self.unet,
            do_spatial_qry_extraction=False,
            do_spatial_post_attn_extraction=self.conf.do_post_attn_injection,
            do_kv_extraction=self.conf.do_pre_attn_injection,
            attend_to_self_kv=self.conf.attend_to_self_kv,
            feature_blend_alpha=self.conf.feature_blend_alpha,
            kv_extraction_paths=self.conf.module_paths,
            spatial_post_attn_extraction_paths=self.conf.module_paths,
            spatial_qry_extraction_paths=[],
            unet_chunk_size=2,
        )
        self.unet.set_attn_processor(self.attn_processor)

        # configure scheduler
        self.scheduler.set_timesteps(self.conf.num_inference_steps)
        n_frames = len(meshes)

        # render depth maps for frames
        depth_maps = render_depth_map(meshes, cameras, 512)

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt] * n_frames)
        stacked_text_embeddings = torch.stack([uncond_embeddings, cond_embeddings])

        # initial latent noise
        latents = self.prepare_latents(meshes, cameras, verts_uvs, faces_uvs, generator)

        # chunk indices to use in inference loop
        chunks_indices = torch.split(torch.arange(0, n_frames), self.conf.chunk_size)

        # precompute visible-vert rasterization for each frame
        vert_xys = []
        vert_indices = []
        for cam, mesh in zip(cameras, meshes):
            xys, idxs = project_visible_verts_to_camera(mesh, cam)
            vert_xys.append(xys)
            vert_indices.append(idxs)

        # denoising loop
        for t in tqdm(self.scheduler.timesteps):
            # update timestep
            self.attn_processor.cur_timestep = t

            # duplicate latent, for classifier-free guidance
            latents_stacked = torch.stack([latents] * 2)
            latents_stacked = self.scheduler.scale_model_input(latents_stacked, t)

            # sample keyframe indices
            kf_indices = self.sample_keyframe_indices(n_frames, generator)

            # Feature Extraction Step
            kf_latents = latents_stacked[:, kf_indices]
            kf_embeddings = stacked_text_embeddings[:, kf_indices]
            kf_depth_maps = [depth_maps[i] for i in kf_indices.tolist()]

            pre_attn_features, post_attn_features = self.model_fwd_feature_extraction(
                kf_latents,
                kf_embeddings,
                kf_depth_maps,
                t,
            )

            # unify spatial features across keyframes as vertex features
            kf_vert_xys = [vert_xys[i] for i in kf_indices.tolist()]
            kf_vert_indices = [vert_indices[i] for i in kf_indices.tolist()]

            aggregated_3d_features = aggregate_spatial_features_dict(
                post_attn_features,
                meshes.num_verts_per_mesh()[0],
                kf_vert_xys,
                kf_vert_indices,
            )

            layer_resolutions = {
                layer: feature.shape[-1]
                for layer, feature in post_attn_features.items()
            }

            # do denoising in chunks
            noise_preds = []
            for chunk_frame_indices in chunks_indices:
                chunk_feature_images = rasterize_and_render_vert_features_dict(
                    aggregated_3d_features,
                    meshes[chunk_frame_indices],
                    cameras[chunk_frame_indices],
                    resolutions=layer_resolutions,
                )

                # Diffusion step #2 with pre and post attn feature injection
                # get chunk inputs
                chunk_latents = latents_stacked[:, chunk_frame_indices]
                chunk_embeddings = stacked_text_embeddings[:, chunk_frame_indices]
                chunk_depth_maps = [depth_maps[i] for i in chunk_frame_indices.tolist()]

                noise_pred = self.model_fwd_feature_injection(
                    chunk_latents,
                    chunk_embeddings,
                    chunk_depth_maps,
                    t,
                    pre_attn_features,
                    chunk_feature_images,
                    chunk_frame_indices,
                )
                noise_preds.append(noise_pred)

            # concatenate predictions
            noise_pred_all = torch.cat(noise_preds, dim=1)

            # preform classifier free guidance
            noise_pred_uncond, noise_pred_cond = noise_pred_all
            noise_pred = noise_pred_uncond + self.conf.guidance_scale * (
                noise_pred_cond - noise_pred_uncond
            )

            # update latents
            latents = self.scheduler.step(
                noise_pred, t, latents, generator=generator
            ).prev_sample

        # decode latents in chunks
        decoded_imgs = []
        for chunk_frame_indices in chunks_indices:
            chunk_latents = latents[chunk_frame_indices]
            chunk_images = self.decode_latents(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        return decoded_imgs
