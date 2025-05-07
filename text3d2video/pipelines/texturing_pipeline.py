from typing import Dict, List, Optional

import torch
from attr import dataclass
from jaxtyping import Float
from PIL import Image
from pytorch3d.renderer.cameras import CamerasBase
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes
from rerun import Tensor
from tqdm import tqdm

from text3d2video.artifacts.anim_artifact import AnimSequence
from text3d2video.attn_processors.final_attn_processor import FinalAttnProcessor
from text3d2video.backprojection import (
    TexelProjection,
    compute_texel_projection,
    project_view_to_texture_masked,
)
from text3d2video.pipelines.controlnet_pipeline import BaseControlNetPipeline
from text3d2video.rendering import (
    compute_autoregressive_update_masks,
    compute_newly_visible_masks,
    compute_uv_jacobian_map,
    downsample_masks,
    make_mesh_rasterizer,
    render_texture,
)
from text3d2video.util import augment_prompt, chunk_dim
from text3d2video.utilities.logging import NULL_LOGGER, setup_greenlists


@dataclass
class TexturingCamSeq:
    cams: CamerasBase
    meshes: Meshes
    verts_uvs: torch.Tensor
    faces_uvs: torch.Tensor
    depth_maps: List[Image.Image]
    projections_hd: List[TexelProjection]
    projections_ld: List[TexelProjection]
    fragments: List[Fragments]
    newly_visible_masks: List[torch.Tensor]
    update_masks: List[torch.Tensor]


@dataclass
class TexGenModelOutput:
    noise_pred: Tensor
    extracted_kvs_uncond: Dict[str, Float[Tensor, "b t d"]]
    extracted_kvs_cond: Dict[str, Float[Tensor, "b t d"]]


class TexturingLogic:
    def __init__(
        self,
        pipe: BaseControlNetPipeline,
        guidance_scale=7.5,
        uv_res=600,
        controlnet_conditioning_scale=1.0,
        quality_factor=1.5,
        module_paths=[],
    ):
        self.pipe = pipe
        self.guidance_scale = guidance_scale
        self.uv_res = uv_res
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.quality_factor = quality_factor

        # create attn processor
        self.attn_processor = FinalAttnProcessor(
            self.pipe.unet,
            do_kv_extraction=True,
            also_attend_to_self=False,
            attend_to_injected=True,
            kv_extraction_paths=module_paths,
        )

    def set_attn_processor(self):
        self.pipe.unet.set_attn_processor(self.attn_processor)

    def model_forward_guided_kvs(
        self,
        latents: Float[Tensor, "b c h w"],
        cond_embeddings: Float[Tensor, "b t d"],
        uncond_embeddings: Float[Tensor, "b d"],
        t: int,
        depth_maps: List[Image.Image],
        injected_kvs_cond: Dict[str, Float[Tensor, "b t d"]] = None,
        injected_kvs_uncond: Dict[str, Float[Tensor, "b d"]] = None,
        extract_kvs: bool = False,
    ) -> TexGenModelOutput:
        latents_duplicated = torch.cat([latents] * 2)
        both_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        depth_maps_duplicated = depth_maps * 2

        # set extraction flag
        self.attn_processor.do_kv_extraction = extract_kvs

        # pass injected kvs
        if injected_kvs_cond is None:
            injected_kvs_cond = {}
        if injected_kvs_uncond is None:
            injected_kvs_uncond = {}
        injected_kvs = {}
        for key in injected_kvs_cond:
            injected_kvs[key] = torch.cat(
                [injected_kvs_uncond[key], injected_kvs_cond[key]]
            )

        self.attn_processor.injected_kvs = injected_kvs

        # do both unet passes
        noise_pred = self.pipe.model_forward(
            latents_duplicated, both_embeddings, t, depth_maps_duplicated
        )

        extracted_kvs = self.attn_processor.extracted_kvs

        # combine predictions according to CFG
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        # TODO deal with guidance scale here
        noise_pred = noise_pred_uncond + self.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        extracted_kvs_cond = {}
        extracted_kvs_uncond = {}
        for key in extracted_kvs:
            kvs = extracted_kvs[key]
            kvs = chunk_dim(kvs, 2)
            extracted_kvs_uncond[key] = kvs[0]
            extracted_kvs_cond[key] = kvs[1]

        return TexGenModelOutput(
            noise_pred=noise_pred,
            extracted_kvs_uncond=extracted_kvs_cond,
            extracted_kvs_cond=extracted_kvs_uncond,
        )

    def compute_newly_visible_masks(self, seq: AnimSequence):
        raster_res = 2000
        projections = []
        for i in range(len(seq)):
            cam = seq.cams[i]
            mesh = seq.meshes[i]

            # project UVs to camera
            projection = compute_texel_projection(
                mesh,
                cam,
                seq.verts_uvs,
                seq.faces_uvs,
                texture_res=self.uv_res,
                raster_res=raster_res,
            )
            projections.append(projection)

        newly_visible_masks = compute_newly_visible_masks(
            seq.cams,
            seq.meshes,
            projections,
            self.uv_res,
            512,
            seq.verts_uvs,
            seq.faces_uvs,
        )

        newly_visible_masks = downsample_masks(
            newly_visible_masks, (64, 64), thresh=0.5
        )

        return newly_visible_masks

    def compute_uv_update_masks(self, seq: AnimSequence):
        projections = [
            compute_texel_projection(
                m, c, seq.verts_uvs, seq.faces_uvs, self.uv_res, raster_res=2000
            )
            for c, m in zip(seq.cams, seq.meshes)
        ]

        # compute quality maps based on image-space gradients
        quality_maps = [
            compute_uv_jacobian_map(c, m, seq.verts_uvs, seq.faces_uvs)
            for c, m in zip(seq.cams, seq.meshes)
        ]
        quality_maps = torch.stack(quality_maps)

        # compute update masks
        better_quality_masks = compute_autoregressive_update_masks(
            seq.cams,
            seq.meshes,
            projections,
            quality_maps,
            self.uv_res,
            seq.verts_uvs,
            seq.faces_uvs,
            quality_factor=self.quality_factor,
        )

        # quality_factor 0: always update
        # quality factor 1.1: update if better quality
        # quality_factor high = newly visible
        return better_quality_masks

    def precompute_cam_seq(self, seq: AnimSequence):
        # newly visible masks
        newly_visible_masks = self.compute_newly_visible_masks(seq)

        # projections and fragments
        PROJ_RASTER_RES = 2000
        PROJ_RASTER_RES_LD = 2000
        projections_hd = []
        projections_ld = []
        fragments = []
        rasterizer = make_mesh_rasterizer(resolution=64)
        for i in range(len(seq)):
            cam = seq.cams[i]
            mesh = seq.meshes[i]

            # project UVs to camera
            projection = compute_texel_projection(
                mesh,
                cam,
                seq.verts_uvs,
                seq.faces_uvs,
                texture_res=self.uv_res,
                raster_res=PROJ_RASTER_RES,
            )
            projections_hd.append(projection)

            projection_ld = compute_texel_projection(
                mesh,
                cam,
                seq.verts_uvs,
                seq.faces_uvs,
                texture_res=self.uv_res,
                raster_res=PROJ_RASTER_RES_LD,
            )
            projections_ld.append(projection_ld)

            # rasterize
            frags = rasterizer(mesh, cameras=seq.cams[i])
            fragments.append(frags)

        # depth maps
        depth_maps = seq.render_depth_maps()

        # update masks
        update_masks = self.compute_uv_update_masks(seq)

        return TexturingCamSeq(
            cams=seq.cams,
            meshes=seq.meshes,
            verts_uvs=seq.verts_uvs,
            faces_uvs=seq.faces_uvs,
            depth_maps=depth_maps,
            projections_hd=projections_hd,
            projections_ld=projections_ld,
            fragments=fragments,
            newly_visible_masks=newly_visible_masks,
            update_masks=update_masks,
        )

    def multiview_denoising(
        self,
        latents,
        cond_embeddings,
        uncond_embeddings,
        t,
        prev_clean_tex: Tensor,
        cam_seq: TexturingCamSeq,
        logger=NULL_LOGGER,
        generator=None,
    ):
        clean_tex = prev_clean_tex.clone()

        verts_uvs = cam_seq.verts_uvs
        faces_uvs = cam_seq.faces_uvs

        blended_latents = []

        ref_kvs_cond = None
        ref_kvs_uncond = None

        n_views = len(cam_seq.cams)
        for i in range(n_views):
            mesh = cam_seq.meshes[i]
            cam = cam_seq.cams[i]
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
            model_out = self.model_forward_guided_kvs(
                blended,
                cond_embeddings[[i]],
                uncond_embeddings[[i]],
                t,
                depth_maps=[depth],
                injected_kvs_cond=ref_kvs_cond,
                injected_kvs_uncond=ref_kvs_uncond,
            )

            noise_pred = model_out.noise_pred

            if i == 0:
                ref_kvs_cond = model_out.extracted_kvs_cond
                ref_kvs_uncond = model_out.extracted_kvs_uncond

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
        cam_seq: TexturingCamSeq,
        generator=None,
    ):
        noise_preds = []

        verts_uvs = cam_seq.verts_uvs
        faces_uvs = cam_seq.faces_uvs

        n_views = len(cam_seq.cams)
        for i in range(n_views):
            mesh = cam_seq.meshes[i]
            cam = cam_seq.cams[i]

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
    module_paths: List[str] = []


class TexturingPipeline(BaseControlNetPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        seq: AnimSequence,
        conf: TexturingConfig,
        texture: Optional[Tensor] = None,
        start_noise: Optional[float] = None,
        generator=None,
        logger=NULL_LOGGER,
    ):
        n_frames = len(seq)

        # make texgen logic object
        method = TexturingLogic(
            self,
            uv_res=conf.uv_res,
            guidance_scale=conf.guidance_scale,
            controlnet_conditioning_scale=conf.controlnet_conditioning_scale,
        )

        # configure scheduler
        self.scheduler.set_timesteps(conf.num_inference_steps)

        setup_greenlists(logger, self.scheduler.timesteps, n_frames, n_save_times=20)

        method.set_attn_processor()

        # Encode Prompt
        prompts = augment_prompt(prompt, n_frames)
        cond_embeddings, uncond_embeddings = self.encode_prompt(prompts)

        # precomputation
        cam_seq = method.precompute_cam_seq(seq)

        logger.write("newly_visible_masks", cam_seq.newly_visible_masks)
        logger.write("update_masks", cam_seq.update_masks)

        use_texture = texture is not None
        if use_texture:
            timesteps = self.get_partial_timesteps(
                conf.num_inference_steps, start_noise
            )
            renders = render_texture(
                seq.meshes, seq.cams, texture, seq.verts_uvs, seq.faces_uvs
            )
            renders_encoded = self.encode_images(renders, generator=generator)
            noise = torch.randn_like(renders_encoded)
            latents = self.scheduler.add_noise(renders_encoded, noise, timesteps[0])

        else:
            timesteps = self.get_partial_timesteps(conf.num_inference_steps, 0)
            latents = self.prepare_latents(len(seq), generator=generator)

        clean_tex = torch.zeros(conf.uv_res, conf.uv_res, 3).cuda()

        # denoising loop
        for i, t in enumerate(tqdm(timesteps)):
            # Stage 1: autoregressively denoise to get denoised texture pred
            updated_latents, clean_tex = method.multiview_denoising(
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

            # update latents
            # latents = updated_latents

            # Stage 2: obtain noise predictions
            rendered_noise_pred = method.render_noise_preds(
                clean_tex,
                latents,
                t,
                cam_seq,
                generator=generator,
            )

            latents_duplicated = torch.cat([latents] * 2)
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

            # 0 at start, 1 at end
            progress = self.noise_variance(t)

            w_texture = w * progress
            w_text = w * (1 - progress)

            noise_pred = (
                noise_uncond
                + w_texture * (noise_tex_cond - noise_uncond)
                + w_text * (noise_cond - noise_uncond)
            )

            # noise_pred = rendered_noise_pred

            # denoise latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents in chunks
        decoded_imgs = []
        chunks_indices = torch.split(torch.arange(0, n_frames), 5)
        for chunk_frame_indices in chunks_indices:
            chunk_latents = latents[chunk_frame_indices]
            chunk_images = self.decode_latents(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        return decoded_imgs
