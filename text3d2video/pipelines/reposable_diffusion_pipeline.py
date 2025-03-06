from typing import Dict, List, Tuple

import torch
from attr import dataclass
from jaxtyping import Float
from PIL import Image
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRenderer,
    join_cameras_as_batch,
)
from pytorch3d.structures import Meshes, join_meshes_as_batch
from torch import Tensor
from tqdm import tqdm

from text3d2video.attn_processors.extraction_injection_attn import (
    ExtractionInjectionAttn,
)
from text3d2video.backprojection import (
    aggregate_views_vert_texture,
    make_repeated_vert_texture,
    project_visible_verts_to_camera,
)
from text3d2video.noise_initialization import NoiseInitializer
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingPipeline,
)
from text3d2video.rendering import TextureShader, make_mesh_rasterizer, render_depth_map
from text3d2video.util import map_dict


@dataclass
class ReposableDiffusionConfig:
    do_pre_attn_injection: bool
    do_post_attn_injection: bool
    aggregate_queries: bool
    feature_blend_alpha: float
    attend_to_self_kv: bool
    mean_features_weight: float
    chunk_size: int
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    module_paths: list[str]


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
        generator=None,
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

        # join meshes and cameras
        all_meshes = join_meshes_as_batch([frame_meshes, aggregation_meshes])
        all_cams = join_cameras_as_batch([frame_cams, aggregation_cams])

        # indices of source and target frames
        source_indices = torch.arange(n_frames, n_frames_total)
        target_indices = torch.arange(n_frames)
        all_indices = torch.arange(n_frames_total)

        # render depth maps
        depth_maps = render_depth_map(all_meshes, all_cams, 512)

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt(
            [prompt] * n_frames_total
        )
        stacked_text_embeddings = torch.stack([uncond_embeddings, cond_embeddings])

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

        source_embeddings = stacked_text_embeddings[:, source_indices]
        source_depth_maps = [depth_maps[i] for i in source_indices.tolist()]
        src_vert_xys = [vert_xys[i] for i in source_indices.tolist()]
        src_vert_indices = [vert_indices[i] for i in source_indices.tolist()]

        # denoising loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
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

            # TODO directly return unstacked
            uncond_spatial_features = map_dict(spatial_features, lambda _, x: x[0])
            cond_spatial_features = map_dict(spatial_features, lambda _, x: x[1])

            layer_resolutions = map_dict(spatial_features, lambda _, x: x.shape[-1])

            def aggregate(layer, features):
                n_verts = aggregation_meshes.num_verts_per_mesh()[0]
                vt_features = aggregate_views_vert_texture(
                    features,
                    n_verts,
                    src_vert_xys,
                    src_vert_indices,
                    mode="nearest",
                    aggregation_type="first",
                )
                return vt_features

            aggregated_cond = map_dict(cond_spatial_features, aggregate)
            aggregated_uncond = map_dict(uncond_spatial_features, aggregate)

            def render_src_frames(layer, texture):
                texture = make_repeated_vert_texture(texture, len(source_indices))

                # make renderer
                rasterizer = make_mesh_rasterizer(
                    cameras=all_cams[source_indices],
                    resolution=layer_resolutions[layer],
                )
                shader = TextureShader()
                renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

                render_meshes = all_meshes[source_indices]
                render_meshes.textures = texture
                return renderer(render_meshes)

            src_rendered_cond = map_dict(aggregated_cond, render_src_frames)
            src_rendered_uncond = map_dict(aggregated_uncond, render_src_frames)

            src_feature_images = {}
            for layer in src_rendered_cond.keys():
                src_feature_images[layer] = torch.stack(
                    [src_rendered_uncond[layer], src_rendered_cond[layer]]
                )

            source_noise_preds = self.model_fwd_feature_injection(
                source_latents,
                source_embeddings,
                source_depth_maps,
                t,
                pre_attn_features,
                src_feature_images,
                source_indices,
            )

            # chunked diffusion step with injection on target frames
            target_noise_preds = []
            for frame_indices in torch.split(target_indices, self.rd_config.chunk_size):
                chunk_cams = all_cams[frame_indices]
                chunk_meshes = all_meshes[frame_indices]

                def render_src_frames(layer, texture):
                    texture = make_repeated_vert_texture(texture, len(chunk_meshes))

                    # make renderer
                    rasterizer = make_mesh_rasterizer(
                        cameras=chunk_cams,
                        resolution=layer_resolutions[layer],
                    )
                    shader = TextureShader()
                    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

                    render_meshes = chunk_meshes
                    render_meshes.textures = texture
                    return renderer(render_meshes)

                chunk_uncond_features = map_dict(aggregated_uncond, render_src_frames)
                chunk_cond_features = map_dict(aggregated_cond, render_src_frames)

                chunk_feature_images = {}
                for layer in src_rendered_cond.keys():
                    chunk_feature_images[layer] = torch.stack(
                        [chunk_uncond_features[layer], chunk_cond_features[layer]]
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
            latents = self.scheduler.step(
                noise_pred, t, latents, generator=generator
            ).prev_sample

        # decode latents in chunks
        decoded_imgs = []
        for chunk_frame_indices in torch.split(all_indices, self.rd_config.chunk_size):
            chunk_latents = latents[chunk_frame_indices]
            chunk_images = self.decode_latents(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        return decoded_imgs
