from typing import List

import torch
from attr import dataclass
from jaxtyping import Float
from PIL import Image
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes
from rerun import Tensor
from tqdm import tqdm

from text3d2video.backprojection import (
    TexelProjection,
    project_view_to_texture_masked,
    project_visible_texels_to_camera,
    update_uv_texture,
)
from text3d2video.pipelines.controlnet_pipeline import BaseControlNetPipeline
from text3d2video.rendering import (
    compute_autoregressive_update_masks,
    compute_newly_visible_masks,
    compute_uv_jacobian_map,
    downsample_masks,
    make_mesh_rasterizer,
    render_depth_map,
    render_texture,
)
from text3d2video.utilities.logging import NULL_LOGGER, setup_greenlists


@dataclass
class CameraSequence:
    cameras: CamerasBase
    meshes: Meshes
    verts_uvs: torch.Tensor
    faces_uvs: torch.Tensor
    depth_maps: List[Image.Image]
    projections_hd: List[TexelProjection]
    projections_ld: List[TexelProjection]
    fragments: List[Fragments]
    newly_visible_masks: List[torch.Tensor]
    update_masks: List[torch.Tensor]


class TexturingLogic:
    def __init__(
        self,
        pipe: BaseControlNetPipeline,
        guidance_scale=7.5,
        uv_res=600,
        controlnet_conditioning_scale=1.0,
    ):
        self.pipe = pipe
        self.guidance_scale = guidance_scale
        self.uv_res = uv_res
        controlnet_conditioning_scale = controlnet_conditioning_scale

    def model_forward_guided(
        self,
        latents: Float[Tensor, "b c h w"],
        cond_embeddings: Float[Tensor, "b t d"],
        uncond_embeddings: Float[Tensor, "b d"],
        t: int,
        depth_maps: List[Image.Image],
    ):
        latents_duplicated = torch.cat([latents] * 2)
        both_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        depth_maps_duplicated = depth_maps * 2

        # do both unet passes
        noise_pred = self.pipe.model_forward(
            latents_duplicated, both_embeddings, t, depth_maps_duplicated
        )

        # combine predictions according to CFG
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        return noise_pred

    def compute_newly_visible_masks(self, cameras, meshes, verts_uvs, faces_uvs):
        raster_res = 2000
        projections = []
        for i in range(len(cameras)):
            cam = cameras[i]
            mesh = meshes[i]

            # project UVs to camera
            projection = project_visible_texels_to_camera(
                mesh,
                cam,
                verts_uvs,
                faces_uvs,
                texture_res=self.uv_res,
                raster_res=raster_res,
            )
            projections.append(projection)

        newly_visible_masks = compute_newly_visible_masks(
            cameras,
            meshes,
            projections,
            self.uv_res,
            512,
            verts_uvs,
            faces_uvs,
        )

        newly_visible_masks = downsample_masks(
            newly_visible_masks, (64, 64), thresh=0.5
        )

        return newly_visible_masks

    def compute_update_masks(self, cameras, meshes, verts_uvs, faces_uvs):
        projections = [
            project_visible_texels_to_camera(
                m, c, verts_uvs, faces_uvs, self.uv_res, raster_res=2000
            )
            for c, m in zip(cameras, meshes)
        ]

        # compute quality maps based on image-space gradients
        quality_maps = [
            compute_uv_jacobian_map(c, m, verts_uvs, faces_uvs)
            for c, m in zip(cameras, meshes)
        ]
        quality_maps = torch.stack(quality_maps)

        # compute update masks
        better_quality_masks = compute_autoregressive_update_masks(
            cameras,
            meshes,
            projections,
            quality_maps,
            self.uv_res,
            verts_uvs,
            faces_uvs,
            quality_factor=1.5,
        )
        return better_quality_masks

    def precompute_cam_seq(self, cameras, meshes, verts_uvs, faces_uvs):
        # newly visible masks
        newly_visible_masks = self.compute_newly_visible_masks(
            cameras, meshes, verts_uvs, faces_uvs
        )

        # projections and fragments
        PROJ_RASTER_RES = 2000
        PROJ_RASTER_RES_LD = 2000
        projections_hd = []
        projections_ld = []
        fragments = []
        rasterizer = make_mesh_rasterizer(resolution=64)
        for i in range(len(cameras)):
            cam = cameras[i]
            mesh = meshes[i]

            # project UVs to camera
            projection = project_visible_texels_to_camera(
                mesh,
                cam,
                verts_uvs,
                faces_uvs,
                texture_res=self.uv_res,
                raster_res=PROJ_RASTER_RES,
            )
            projections_hd.append(projection)

            projection_ld = project_visible_texels_to_camera(
                mesh,
                cam,
                verts_uvs,
                faces_uvs,
                texture_res=self.uv_res,
                raster_res=PROJ_RASTER_RES_LD,
            )
            projections_ld.append(projection_ld)

            # rasterize
            frags = rasterizer(mesh, cameras=cameras[i])
            fragments.append(frags)

        # depth maps
        depth_maps = render_depth_map(meshes, cameras)

        # update masks
        update_masks = self.compute_update_masks(cameras, meshes, verts_uvs, faces_uvs)

        return CameraSequence(
            cameras=cameras,
            meshes=meshes,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
            depth_maps=depth_maps,
            projections_hd=projections_hd,
            projections_ld=projections_ld,
            fragments=fragments,
            newly_visible_masks=newly_visible_masks,
            update_masks=update_masks,
        )

    def predict_clean_texture(
        self,
        latents,
        cond_embeddings,
        uncond_embeddings,
        t,
        prev_clean_tex: Tensor,
        cam_seq: CameraSequence,
        logger=NULL_LOGGER,
        generator=None,
    ):
        clean_tex = prev_clean_tex.clone()

        verts_uvs = cam_seq.verts_uvs
        faces_uvs = cam_seq.faces_uvs

        blended_latents = []

        n_views = len(cam_seq.cameras)
        for i in range(n_views):
            mesh = cam_seq.meshes[i]
            cam = cam_seq.cameras[i]
            proj = cam_seq.projections_ld[i]
            depth = cam_seq.depth_maps[i]
            latent = latents[i]
            mask_i = cam_seq.newly_visible_masks[i].cuda()

            # render partial texture
            rendered = render_texture(mesh, cam, clean_tex, verts_uvs, faces_uvs)
            logger.write("rendered", rendered[0], t=t, frame_i=i)

            # bring render to noise level
            rendered_latent = self.pipe.encode_images(rendered)
            noise = torch.randn_like(rendered_latent)
            rendered_noisy = self.pipe.scheduler.add_noise(rendered_latent, noise, t)[0]

            # blend according to mask
            blended = latent * mask_i + rendered_noisy * (1 - mask_i)
            blended = blended.to(latent)
            blended = blended.unsqueeze(0)

            if i == 0:
                blended = latent.unsqueeze(0)

            # noise pred
            noise_pred = self.model_forward_guided(
                blended,
                cond_embeddings[[i]],
                uncond_embeddings[[i]],
                t,
                depth_maps=[depth],
            )

            # get clean image
            clean_im = self.pipe.scheduler.step(
                noise_pred, t, blended
            ).pred_original_sample
            clean_im_rgb = self.pipe.decode_latents(
                clean_im, output_type="pt", generator=generator
            )[0]

            logger.write("clean_im", clean_im_rgb, t=t, frame_i=i)

            update_mask = cam_seq.update_masks[i].cuda()
            project_view_to_texture_masked(
                clean_tex,
                clean_im_rgb,
                update_mask,
                proj,
            )

            blended_latents.append(blended[0])

        blended_latents = torch.stack(blended_latents, dim=0)

        return blended_latents, clean_tex

    def render_noise_preds(
        self,
        clean_tex: Tensor,
        latents,
        t,
        cam_seq: CameraSequence,
        generator=None,
    ):
        noise_preds = []

        verts_uvs = cam_seq.verts_uvs
        faces_uvs = cam_seq.faces_uvs

        n_views = len(cam_seq.cameras)
        for i in range(n_views):
            mesh = cam_seq.meshes[i]
            cam = cam_seq.cameras[i]

            # render view-consistent denoised observation for view i
            rendered = render_texture(mesh, cam, clean_tex, verts_uvs, faces_uvs)[0]
            rendered_latent = self.pipe.encode_images([rendered], generator)[0]

            # convert denoised observation to noise prediction
            alpha_cumprod_t = self.pipe.scheduler.alphas_cumprod[t]
            sqrt_alpha = alpha_cumprod_t.sqrt()
            sqrt_one_minus_alpha = (1.0 - alpha_cumprod_t).sqrt()

            noisy = latents[i]
            noise_pred_render = (
                noisy - sqrt_alpha * rendered_latent
            ) / sqrt_one_minus_alpha

            noise_preds.append(noise_pred_render)

        noise_preds = torch.stack(noise_preds, dim=0)
        return noise_preds


@dataclass
class TexturingConfig:
    num_inference_steps: int = 15
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    uv_res: int = 600


class TexturingPipeline(BaseControlNetPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        meshes: Meshes,
        cameras: CamerasBase,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        texgen_config: TexturingConfig,
        prompt_suffixes: List[str] = None,
        generator=None,
        logger=NULL_LOGGER,
    ):
        n_frames = len(meshes)

        # make texgen logic object
        method = TexturingLogic(
            self,
            uv_res=texgen_config.uv_res,
            guidance_scale=texgen_config.guidance_scale,
            controlnet_conditioning_scale=texgen_config.controlnet_conditioning_scale,
        )

        # configure scheduler
        self.scheduler.set_timesteps(texgen_config.num_inference_steps)

        setup_greenlists(logger, self.scheduler.timesteps, n_frames, n_save_times=20)

        # augment prompts
        prompts = [prompt] * n_frames
        if prompt_suffixes is not None:
            prompts = [p + s for p, s in zip(prompts, prompt_suffixes)]

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt(prompts)

        # precomputation
        cam_seq = method.precompute_cam_seq(cameras, meshes, verts_uvs, faces_uvs)

        # initial latent noise
        latents = self.prepare_latents(len(meshes), generator=generator)

        clean_tex = torch.zeros(600, 600, 3).cuda()

        # denoising loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            # Stage 1: autoregressively denoise to get denoised texture pred
            updated_latents, clean_tex = method.predict_clean_texture(
                latents,
                cond_embeddings,
                uncond_embeddings,
                t,
                clean_tex,
                cam_seq,
                logger,
                generator=generator,
            )
            logger.write("clean_tex", clean_tex, t=t)

            # Stage 2: obtain noise predictions
            rendered_noise_pred = method.render_noise_preds(
                clean_tex,
                updated_latents,
                t,
                cam_seq,
                generator=generator,
            )

            latents_duplicated = torch.cat([updated_latents] * 2)
            both_embeddings = torch.cat([cond_embeddings, uncond_embeddings])

            noise_pred = self.model_forward(
                latents_duplicated,
                both_embeddings,
                t,
                cam_seq.depth_maps * 2,
            )

            noise_cond, noise_uncond = noise_pred.chunk(2)

            w = 7.5
            noise_tex_cond = 1 / w * (rendered_noise_pred - noise_uncond) + noise_uncond

            # 0 in first iteration, 1 in last
            # progress = self.denoising_progress(t)
            # 0 at start, 1 at end
            progress = self.noise_variance(t)
            progress = 1 - progress

            w_texture = w * (1 - progress)
            w_text = w * (progress)

            print("texture", w_texture, "text", w_text)

            noise_pred = (
                noise_uncond
                + w_texture * (noise_tex_cond - noise_uncond)
                + w_text * (noise_cond - noise_uncond)
            )

            # denoise latents
            latents = self.scheduler.step(noise_pred, t, updated_latents).prev_sample

        # decode latents in chunks
        decoded_imgs = []
        chunks_indices = torch.split(torch.arange(0, n_frames), 5)
        for chunk_frame_indices in chunks_indices:
            chunk_latents = latents[chunk_frame_indices]
            chunk_images = self.decode_latents(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        return decoded_imgs
