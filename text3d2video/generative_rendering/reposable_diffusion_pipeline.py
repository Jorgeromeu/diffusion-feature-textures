from typing import Dict, List, Tuple

import torch
from jaxtyping import Float
from PIL import Image
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    join_cameras_as_batch,
)
from pytorch3d.structures import Meshes, join_meshes_as_batch
from torch import Tensor
from tqdm import tqdm

from text3d2video.artifacts.gr_data import GrDataArtifact, GrSaveConfig
from text3d2video.backprojection import (
    aggregate_spatial_features_dict,
    project_visible_verts_to_cameras,
    rasterize_and_render_vert_features_dict,
)
from text3d2video.generative_rendering.configs import (
    ReposableDiffusionConfig,
)
from text3d2video.generative_rendering.extraction_injection_attn import (
    ExtractionInjectionAttn,
)
from text3d2video.generative_rendering.generative_rendering_pipeline import (
    GenerativeRenderingPipeline,
)
from text3d2video.noise_initialization import NoiseInitializer
from text3d2video.rendering import render_depth_map


class ReposableDiffusionPipeline(GenerativeRenderingPipeline):
    rd_config: ReposableDiffusionConfig
    attn_processor: ExtractionInjectionAttn

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
        # set extraction mode
        self.attn_processor.set_extraction_mode()

        # forward pass
        self.model_forward(latents, text_embeddings, t, depth_maps=depth_maps)

        # read features
        pre_attn_features = self.attn_processor.kv_features

        if self.rd_config.aggregate_queries:
            spatial_features = self.attn_processor.spatial_qry_features
        else:
            spatial_features = self.attn_processor.spatial_post_attn_features

        self.attn_processor.clear_features()

        return pre_attn_features, spatial_features

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
        if self.rd_config.aggregate_queries:
            injected_qrys = feature_images
            injected_post_attn = {}
        else:
            injected_qrys = {}
            injected_post_attn = feature_images

        self.attn_processor.set_injection_mode(
            pre_attn_features=pre_attn_features,
            post_attn_features=injected_post_attn,
            qry_features=injected_qrys,
        )

        # pass frame indices to attn processor and timestep to attn processor
        self.attn_processor.set_chunk_frame_indices(frame_indices)

        noise_pred = self.model_forward(
            latents, text_embeddings, t, depth_maps=depth_maps
        )

        return noise_pred

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        frame_meshes: Meshes,
        frame_cams: FoVPerspectiveCameras,
        aggregation_meshes: Meshes,
        aggregation_cams: FoVPerspectiveCameras,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        reposable_diffusion_config: ReposableDiffusionConfig,
        noise_initializer: NoiseInitializer,
        gr_save_config: GrSaveConfig,
    ):
        # setup configs for use throughout pipeline
        self.rd_config = reposable_diffusion_config
        self.noise_initializer = noise_initializer

        if self.rd_config.aggregate_queries:
            do_post_attn_extraction = False
            do_qry_extraction = self.rd_config.do_post_attn_injection
            qry_extraction_paths = self.rd_config.module_paths
            post_attn_extraction_paths = []
        else:
            do_post_attn_extraction = self.rd_config.do_post_attn_injection
            do_qry_extraction = False
            qry_extraction_paths = []
            post_attn_extraction_paths = self.rd_config.module_paths

        # setup attn processor
        self.attn_processor = ExtractionInjectionAttn(
            self.unet,
            do_spatial_qry_extraction=do_qry_extraction,
            do_spatial_post_attn_extraction=do_post_attn_extraction,
            do_kv_extraction=self.rd_config.do_pre_attn_injection,
            attend_to_self_kv=self.rd_config.attend_to_self_kv,
            feature_blend_alpha=self.rd_config.feature_blend_alpha,
            kv_extraction_paths=self.rd_config.module_paths,
            spatial_qry_extraction_paths=qry_extraction_paths,
            spatial_post_attn_extraction_paths=post_attn_extraction_paths,
        )

        self.unet.set_attn_processor(self.attn_processor)

        # configure scheduler
        self.scheduler.set_timesteps(self.rd_config.num_inference_steps)
        n_frames = len(frame_meshes)
        n_aggr_frames = len(aggregation_meshes)
        n_frames_total = n_frames + n_aggr_frames

        # setup data artifact
        data_artifact = GrDataArtifact.init_from_config(gr_save_config)
        self.gr_data_artifact = data_artifact
        self.attn_processor.set_attn_data_writer(data_artifact.attn_writer)
        self.gr_data_artifact.begin_recording(self.scheduler, n_frames)

        # setup generator
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.rd_config.seed)

        # join meshes and cameras
        all_meshes = join_meshes_as_batch([frame_meshes, aggregation_meshes])
        all_cams = join_cameras_as_batch([frame_cams, aggregation_cams])

        # indices of source and target frames
        source_indices = torch.arange(n_frames, n_frames_total)
        target_indices = torch.arange(n_frames)
        all_indices = torch.arange(n_frames_total)

        # render depth maps
        depth_maps = render_depth_map(all_meshes, all_cams, self.rd_config.resolution)

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt(
            [prompt] * n_frames_total
        )
        stacked_text_embeddings = torch.stack([uncond_embeddings, cond_embeddings])

        # initial noise
        latents = self.prepare_latents(
            all_meshes, all_cams, verts_uvs, faces_uvs, generator
        )

        # get 2D vertex positions for each frame
        vert_xys, vert_indices = project_visible_verts_to_cameras(all_meshes, all_cams)

        source_embeddings = stacked_text_embeddings[:, source_indices]
        source_depth_maps = [depth_maps[i] for i in source_indices.tolist()]
        src_vert_xys = [vert_xys[i] for i in source_indices.tolist()]
        src_vert_indices = [vert_indices[i] for i in source_indices.tolist()]

        # denoising loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            self.gr_data_artifact.latents_writer.write_latents_batched(t, latents)

            # set attn processor timestep
            self.attn_processor.set_cur_timestep(t)

            # duplicate latents, for classifier-free guidance
            latents_stacked = torch.stack([latents] * 2)
            latents_stacked = self.scheduler.scale_model_input(latents_stacked, t)

            # Diffusion step on source frames, to extract features
            source_latents = latents_stacked[:, source_indices]
            pre_attn_features, spatial_features = self.model_fwd_feature_extraction(
                source_latents,
                source_embeddings,
                source_depth_maps,
                t,
            )

            # aggregate spatial features
            aggregated_3d_features = aggregate_spatial_features_dict(
                spatial_features,
                aggregation_meshes.num_verts_per_mesh()[0],
                src_vert_xys,
                src_vert_indices,
            )

            layer_resolutions = {
                layer: feature.shape[-1] for layer, feature in spatial_features.items()
            }

            # save aggregated features
            self.gr_data_artifact.gr_writer.write_vertex_features(
                t, aggregated_3d_features
            )

            # injection step on source frames
            source_feature_images = rasterize_and_render_vert_features_dict(
                aggregated_3d_features,
                all_meshes[source_indices],
                all_cams[source_indices],
                resolutions=layer_resolutions,
            )

            source_noise_preds = self.model_fwd_feature_injection(
                source_latents,
                source_embeddings,
                source_depth_maps,
                t,
                pre_attn_features,
                source_feature_images,
                source_indices,
            )

            # chunked diffusion step with injection on target frames
            target_noise_preds = []
            for frame_indices in torch.split(target_indices, self.rd_config.chunk_size):
                chunk_cams = all_cams[frame_indices]
                chunk_meshes = all_meshes[frame_indices]

                chunk_feature_images = rasterize_and_render_vert_features_dict(
                    aggregated_3d_features,
                    chunk_meshes,
                    chunk_cams,
                    resolutions=layer_resolutions,
                )

                chunk_latents = latents_stacked[:, frame_indices]
                chunk_embeddings = stacked_text_embeddings[:, frame_indices]
                chunk_depth_maps = [depth_maps[i] for i in frame_indices.tolist()]

                if not self.rd_config.do_pre_attn_injection:
                    pre_attn_features = {}
                if not self.rd_config.do_post_attn_injection:
                    chunk_feature_images = {}

                noise_pred = self.model_fwd_feature_injection(
                    chunk_latents,
                    chunk_embeddings,
                    chunk_depth_maps,
                    t,
                    pre_attn_features,
                    chunk_feature_images,
                    frame_indices,
                )
                target_noise_preds.append(noise_pred)
            target_noise_preds = torch.cat(target_noise_preds, dim=1)

            noise_pred_all = torch.cat([target_noise_preds, source_noise_preds], dim=1)
            torch.cuda.empty_cache()

            # preform classifier free guidance
            noise_pred_uncond, noise_pred_text = noise_pred_all
            guidance_direction = noise_pred_text - noise_pred_uncond
            noise_pred = (
                noise_pred_uncond + self.rd_config.guidance_scale * guidance_direction
            )

            # update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        self.gr_data_artifact.latents_writer.write_latents_batched(0, latents)

        # decode latents in chunks
        decoded_imgs = []
        for chunk_frame_indices in torch.split(all_indices, self.rd_config.chunk_size):
            chunk_latents = latents[chunk_frame_indices]
            chunk_images = self.decode_latents(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        self.gr_data_artifact.end_recording()
        return decoded_imgs
