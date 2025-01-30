import torch
from pytorch3d.renderer import FoVPerspectiveCameras, join_cameras_as_batch
from pytorch3d.structures import Meshes, join_meshes_as_batch
from tqdm import tqdm

from text3d2video.artifacts.gr_data import GrDataArtifact, GrSaveConfig
from text3d2video.backprojection import project_vertices_to_cameras
from text3d2video.generative_rendering.configs import (
    ReposableDiffusionConfig,
)
from text3d2video.generative_rendering.generative_rendering_attn import (
    GenerativeRenderingAttn,
)
from text3d2video.generative_rendering.generative_rendering_pipeline import (
    GenerativeRenderingPipeline,
)
from text3d2video.noise_initialization import NoiseInitializer
from text3d2video.rendering import render_depth_map


class ReposableDiffusionPipeline(GenerativeRenderingPipeline):
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
        self.gr_config = reposable_diffusion_config
        self.noise_initializer = noise_initializer

        # set up attention processor
        self.attn_processor = GenerativeRenderingAttn(
            self.unet, self.gr_config, unet_chunk_size=2
        )
        self.unet.set_attn_processor(self.attn_processor)

        # configure scheduler
        self.scheduler.set_timesteps(self.gr_config.num_inference_steps)
        n_frames = len(frame_meshes)
        n_aggr_frames = len(aggregation_meshes)
        n_frames_total = n_frames + n_aggr_frames

        # setup diffusion data
        data_artifact = GrDataArtifact.init_from_config(gr_save_config)
        self.gr_data_artifact = data_artifact
        self.attn_processor.gr_data_artifact = data_artifact
        self.gr_data_artifact.begin_recording(self.scheduler, n_frames)

        # setup generator
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.gr_config.seed)

        # join meshes and cameras
        all_meshes = join_meshes_as_batch([frame_meshes, aggregation_meshes])
        all_cams = join_cameras_as_batch([frame_cams, aggregation_cams])
        all_indices = torch.arange(n_frames_total)
        aggr_indices = torch.arange(n_frames, n_frames_total)

        # render depth maps
        depth_maps = render_depth_map(all_meshes, all_cams, self.gr_config.resolution)

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt(
            [prompt] * n_frames_total
        )
        stacked_text_embeddings = torch.stack([uncond_embeddings, cond_embeddings])

        # initial noise
        latents = self.prepare_latents(
            all_meshes, all_cams, verts_uvs, faces_uvs, generator
        )

        # indices to use in chunked inference loops
        chunk_indices = torch.split(all_indices, self.gr_config.chunk_size)

        # get 2D vertex positions for each frame
        vert_xys, vert_indices = project_vertices_to_cameras(all_meshes, all_cams)

        # denoising loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            self.gr_data_artifact.latents_writer.write_latents_batched(t, latents)

            # update timestep
            self.attn_processor.cur_timestep = t

            # duplicate latents, for classifier-free guidance
            latents_stacked = torch.stack([latents] * 2)
            latents_stacked = self.scheduler.scale_model_input(latents_stacked, t)

            # Diffusion step #1 on keyframes, to extract features
            aggr_latents = latents_stacked[:, aggr_indices]
            aggr_embeddings = stacked_text_embeddings[:, aggr_indices]
            aggr_depth_maps = [depth_maps[i] for i in aggr_indices.tolist()]

            pre_attn_features, post_attn_features = (
                self.model_forward_feature_extraction(
                    aggr_latents,
                    aggr_embeddings,
                    aggr_depth_maps,
                    t,
                )
            )

            # save kf post attn features and indices
            # self.gr_data_artifact.gr_writer.write_kf_indices(t, kf_indices)
            # self.gr_data_artifact.gr_writer.write_kf_post_attn(t, post_attn_features)

            # unify spatial features across keyframes as vertex features
            aggr_vert_xys = [vert_xys[i] for i in aggr_indices.tolist()]
            aggr_vert_indices = [vert_indices[i] for i in aggr_indices.tolist()]
            aggregated_3d_features = self.aggregate_feature_maps(
                frame_meshes.num_verts_per_mesh()[0],
                aggr_vert_xys,
                aggr_vert_indices,
                post_attn_features,
            )

            # save aggregated features
            self.gr_data_artifact.gr_writer.write_vertex_features(
                t, aggregated_3d_features
            )

            # do inference in chunks
            noise_preds = []
            for chunk_frame_indices in chunk_indices:
                # render chunk feature images
                chunk_feature_images = self.render_feature_images(
                    all_cams[chunk_frame_indices],
                    all_meshes[chunk_frame_indices],
                    aggregated_3d_features,
                )

                # Diffusion step #2 with pre and post attn feature injection
                # get chunk inputs
                chunk_latents = latents_stacked[:, chunk_frame_indices]
                chunk_embeddings = stacked_text_embeddings[:, chunk_frame_indices]
                chunk_depth_maps = [depth_maps[i] for i in chunk_frame_indices.tolist()]

                noise_pred = self.model_forward_feature_injection(
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
            noise_pred_uncond, noise_pred_text = noise_pred_all
            guidance_direction = noise_pred_text - noise_pred_uncond
            noise_pred = (
                noise_pred_uncond + self.gr_config.guidance_scale * guidance_direction
            )

            # update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # self.gr_data_artifact.latents_writer.write_latents_batched(0, latents)

        # decode latents in chunks
        decoded_imgs = []
        for chunk_frame_indices in chunk_indices:
            chunk_latents = latents[chunk_frame_indices]
            chunk_images = self.decode_latents(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        self.gr_data_artifact.end_recording()
        return decoded_imgs
