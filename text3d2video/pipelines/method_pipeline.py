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

from text3d2video.attn_processors.final_attn_processor import FinalAttnProcessor
from text3d2video.backprojection import (
    project_visible_texels_to_camera,
)
from text3d2video.pipelines.controlnet_pipeline import BaseControlNetPipeline
from text3d2video.pipelines.texgen_pipeline import TexGenLogic
from text3d2video.rendering import (
    compute_autoregressive_update_masks,
    compute_newly_visible_masks,
    compute_uv_jacobian_map,
    downsample_masks,
    make_mesh_rasterizer,
    render_depth_map,
    render_texture,
    shade_mesh,
)
from text3d2video.util import chunk_dim, sample_feature_map_ndc
from text3d2video.utilities.logging import GrLogger, H5Logger


@dataclass
class MethodConfig:
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    module_paths: List[str]
    quality_update_factor: float
    uv_res: int
    t_threshold: float


@dataclass
class TexGenModelOutput:
    noise_pred: Tensor
    extracted_kvs_uncond: Dict[str, Float[Tensor, "b t d"]]
    extracted_kvs_cond: Dict[str, Float[Tensor, "b t d"]]


class MethodPipeline(BaseControlNetPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        tex_meshes: Meshes,
        tex_cameras: FoVPerspectiveCameras,
        tex_verts_uvs: torch.Tensor,
        tex_faces_uvs: torch.Tensor,
        anim_meshes: Meshes,
        anim_cameras: FoVPerspectiveCameras,
        anim_verts_uvs: torch.Tensor,
        anim_faces_uvs: torch.Tensor,
        conf: MethodConfig,
        prompt_suffixes: List[str] = None,
        generator=None,
        logger=None,
    ):
        # make texgen logic object
        texgen = TexGenLogic(
            self,
            conf.uv_res,
            conf.guidance_scale,
            conf.controlnet_conditioning_scale,
            conf.module_paths,
        )

        n_mv_frames = len(tex_meshes)
        n_anim_frames = len(anim_meshes)

        # configure scheduler
        self.scheduler.set_timesteps(conf.num_inference_steps)

        # setup logger
        if logger is None:
            logger = H5Logger.create_disabled()

        # augment prompts
        prompts = [prompt] * n_mv_frames
        if prompt_suffixes is not None:
            prompts = [p + s for p, s in zip(prompts, prompt_suffixes)]

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt(prompts)
        cond_embeddings_anim, uncond_embeddings_anim = self.encode_prompt(
            [prompt] * n_anim_frames
        )

        # render depth maps for frames
        depth_maps = render_depth_map(tex_meshes, tex_cameras, 512)
        anim_depth_maps = render_depth_map(anim_meshes, anim_cameras, 512)

        # initial latent noise
        mv_latents = self.prepare_latents(len(tex_meshes), generator=generator)
        anim_latents = self.prepare_latents(len(anim_meshes), generator=generator)

        # TODO move constants
        image_res = 512

        # precompute rasterization and projections for mv frames
        projections = []
        fragments = []
        rasterizer = make_mesh_rasterizer(resolution=image_res)
        for i in range(n_mv_frames):
            cam = tex_cameras[i]
            mesh = tex_meshes[i]

            # project UVs to camera
            projection = project_visible_texels_to_camera(
                mesh,
                cam,
                tex_verts_uvs,
                tex_faces_uvs,
                texture_res=conf.uv_res,
                raster_res=10000,
            )
            projections.append(projection)

            # rasterize
            frags = rasterizer(tex_meshes[i], cameras=tex_cameras[i])
            fragments.append(frags)

        # precompute rasterization for anim frames
        anim_fragments = []
        for i in range(len(anim_meshes)):
            cam = anim_cameras[i]
            mesh = anim_meshes[i]

            frags = rasterizer(anim_meshes[i], cameras=anim_cameras[i])
            anim_fragments.append(frags)

        # precompute newly-visible masks
        newly_visible_masks = compute_newly_visible_masks(
            tex_cameras,
            tex_meshes,
            projections,
            conf.uv_res,
            image_res,
            tex_verts_uvs,
            tex_faces_uvs,
        )
        newly_visible_masks_down = downsample_masks(
            newly_visible_masks, (64, 64), thresh=0.1
        ).cuda()

        # compute quality maps
        quality_maps = [
            compute_uv_jacobian_map(c, m, tex_verts_uvs, tex_faces_uvs)
            for c, m in zip(tex_cameras, tex_meshes)
        ]
        quality_maps = torch.stack(quality_maps)

        better_quality_masks = compute_autoregressive_update_masks(
            tex_cameras,
            tex_meshes,
            projections,
            quality_maps,
            conf.uv_res,
            tex_verts_uvs,
            tex_faces_uvs,
            quality_factor=conf.quality_update_factor,
        )

        # denoising loop
        for t in tqdm(self.scheduler.timesteps):
            # Autoregressively denoise texturing views to obtain clean texture prediction
            clean_tex = texgen.predict_clean_texture(
                mv_latents,
                cond_embeddings,
                uncond_embeddings,
                t,
                tex_meshes,
                tex_verts_uvs,
                tex_faces_uvs,
                projections,
                fragments,
                depth_maps,
                newly_visible_masks_down,
                better_quality_masks,
                generator=generator,
            )

            # Render noise preds for texturing views
            noise_preds = texgen.render_noise_preds(
                clean_tex,
                tex_meshes,
                tex_cameras,
                tex_verts_uvs,
                tex_faces_uvs,
                mv_latents,
                t,
                generator=generator,
            )

            if t > conf.t_threshold:
                # obtain noise preds from animation
                noise_preds_anim = texgen.render_noise_preds(
                    clean_tex,
                    anim_meshes,
                    anim_cameras,
                    anim_verts_uvs,
                    anim_faces_uvs,
                    anim_latents,
                    t,
                    generator=generator,
                )
            else:
                print("ddim_step")
                noise_preds_anim = self.model_forward_cfg(
                    anim_latents,
                    cond_embeddings_anim,
                    uncond_embeddings_anim,
                    t,
                    anim_depth_maps,
                    controlnet_conditioning_scale=conf.controlnet_conditioning_scale,
                    guidance_scale=conf.guidance_scale,
                )

            # denoise latents
            mv_latents = self.scheduler.step(noise_preds, t, mv_latents).prev_sample
            anim_latents = self.scheduler.step(
                noise_preds_anim, t, anim_latents
            ).prev_sample

        # decode mv latents in chunks
        decoded_imgs = []
        chunks_indices = torch.split(torch.arange(0, n_mv_frames), 5)
        for chunk_frame_indices in chunks_indices:
            chunk_latents = mv_latents[chunk_frame_indices]
            chunk_images = self.decode_latents(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        # decode anim latents in chunks
        decoded_imgs_anim = []
        chunks_indices = torch.split(torch.arange(0, n_anim_frames), 5)
        for chunk_frame_indices in chunks_indices:
            chunk_latents = anim_latents[chunk_frame_indices]
            chunk_images = self.decode_latents(chunk_latents, generator)
            decoded_imgs_anim.extend(chunk_images)

        return decoded_imgs, decoded_imgs_anim
