import torch
from attr import dataclass
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    join_cameras_as_batch,
)
from pytorch3d.structures import Meshes, join_meshes_as_batch
from tqdm import tqdm

from text3d2video.attn_processors.extraction_injection_attn import (
    ExtractionInjectionAttn,
)
from text3d2video.backprojection import (
    project_visible_verts_to_camera,
)
from text3d2video.noise_initialization import NoiseInitializer
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingPipeline,
)
from text3d2video.rendering import (
    render_depth_map,
)
from text3d2video.util import dict_map


@dataclass
class ReposableDiffusionConfig:
    do_pre_attn_injection: bool
    do_post_attn_injection: bool
    feature_blend_alpha: float
    attend_to_self_kv: bool
    mean_features_weight: float
    chunk_size: int
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    module_paths: list[str]


class ReposableDiffusionPipeline(GenerativeRenderingPipeline):
    conf: ReposableDiffusionConfig
    attn_processor: ExtractionInjectionAttn

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        tgt_meshes: Meshes,
        tgt_cams: FoVPerspectiveCameras,
        src_meshes: Meshes,
        src_cams: FoVPerspectiveCameras,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        reposable_diffusion_config: ReposableDiffusionConfig,
        noise_initializer: NoiseInitializer,
        generator=None,
    ):
        # setup configs for use throughout pipeline
        self.conf = reposable_diffusion_config
        self.noise_initializer = noise_initializer

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
        n_tgt_frames = len(tgt_meshes)
        n_src_frames = len(src_meshes)
        n_frames_total = n_tgt_frames + n_src_frames

        # join meshes and cameras
        all_meshes = join_meshes_as_batch([tgt_meshes, src_meshes])
        all_cams = join_cameras_as_batch([tgt_cams, src_cams])

        # indices of source and target frames
        src_indices = torch.arange(n_tgt_frames, n_frames_total)
        tgt_indices = torch.arange(n_tgt_frames)
        all_indices = torch.arange(n_frames_total)

        # render all depth maps
        all_depth_maps = render_depth_map(all_meshes, all_cams, 512)

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt(
            [prompt] * n_frames_total
        )

        # initial noise
        latents = self.prepare_latents(
            all_meshes, all_cams, verts_uvs, faces_uvs, generator
        )

        # precompute visible-vert rasterization for each frame
        vert_xys = []
        vert_indices = []
        for cam, mesh in zip(all_cams, all_meshes):
            xys, idxs = project_visible_verts_to_camera(mesh, cam)
            vert_xys.append(xys)
            vert_indices.append(idxs)

        src_uncond_embeddings = uncond_embeddings[src_indices]
        src_cond_embeddings = cond_embeddings[src_indices]
        src_depth_maps = [all_depth_maps[i] for i in src_indices.tolist()]

        # denoising loop
        for t in tqdm(self.scheduler.timesteps):
            # set attn processor timestep
            self.attn_processor.set_cur_timestep(t)

            # Denoise source frames with extraction
            features = self.model_forward_extraction(
                latents[src_indices],
                src_cond_embeddings,
                src_uncond_embeddings,
                src_depth_maps,
                t,
            )

            src_vert_xys = [vert_xys[i] for i in src_indices.tolist()]
            src_vert_indices = [vert_indices[i] for i in src_indices.tolist()]

            def aggr_src_features(_, features):
                return self.aggregate_features(
                    features,
                    tgt_meshes.num_verts_per_mesh()[0],
                    src_vert_xys,
                    src_vert_indices,
                )

            textures_uncond = dict_map(features.uncond_post_attn, aggr_src_features)
            textures_cond = dict_map(features.cond_post_attn, aggr_src_features)

            def render_src_frames(layer, texture):
                return self.render_texture(
                    texture, src_cams, src_meshes, features.layer_resolution(layer)
                )

            src_rendered_uncond = dict_map(textures_uncond, render_src_frames)
            src_rendered_cond = dict_map(textures_cond, render_src_frames)

            # Denoising source frames with injection
            src_noise_preds = self.model_forward_injection(
                latents[src_indices],
                cond_embeddings[src_indices],
                uncond_embeddings[src_indices],
                [all_depth_maps[i] for i in src_indices.tolist()],
                t,
                features.cond_kv,
                features.uncond_kv,
                src_rendered_cond,
                src_rendered_uncond,
            )

            # Denoise target frames with injection
            tgt_noise_preds = []
            for chunk_frame_indices in torch.split(tgt_indices, self.conf.chunk_size):

                def render_chunk_frames(layer, texture):
                    chunk_cams = all_cams[chunk_frame_indices]
                    chunk_meshes = all_meshes[chunk_frame_indices]
                    return self.render_texture(
                        texture,
                        chunk_cams,
                        chunk_meshes,
                        features.layer_resolution(layer),
                    )

                chunk_rendered_uncond = dict_map(textures_uncond, render_chunk_frames)
                chunk_rendered_cond = dict_map(textures_cond, render_chunk_frames)

                chunk_noise = self.model_forward_injection(
                    latents[chunk_frame_indices],
                    cond_embeddings[chunk_frame_indices],
                    uncond_embeddings[chunk_frame_indices],
                    [all_depth_maps[i] for i in chunk_frame_indices.tolist()],
                    t,
                    features.cond_kv,
                    features.uncond_kv,
                    chunk_rendered_cond,
                    chunk_rendered_uncond,
                )

                tgt_noise_preds.append(chunk_noise)

            # Denoise all
            tgt_noise_preds = torch.cat(tgt_noise_preds)
            all_noise_preds = torch.cat([tgt_noise_preds, src_noise_preds])
            latents = self.scheduler.step(
                all_noise_preds, t, latents, generator=generator
            ).prev_sample

        # decode latents in chunks
        decoded_imgs = []
        for chunk_frame_indices in torch.split(all_indices, self.conf.chunk_size):
            chunk_latents = latents[chunk_frame_indices]
            chunk_images = self.decode_latents(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        return decoded_imgs
