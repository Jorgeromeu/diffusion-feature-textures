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
    compute_newly_visible_masks,
    compute_texel_projection,
    compute_texel_projections,
    project_view_to_texture_masked,
    update_uv_texture,
)
from text3d2video.pipelines.controlnet_pipeline import BaseControlNetPipeline
from text3d2video.rendering import (
    compute_autoregressive_update_masks,
    compute_uv_jacobian_map,
    downsample_masks,
    make_mesh_rasterizer,
    render_texture,
)
from text3d2video.sd_feature_extraction import AttnType, BlockType, find_attn_modules
from text3d2video.util import augment_prompt, chunk_dim
from text3d2video.utilities.logging import (
    NULL_LOGGER,
    setup_greenlists,
)


@dataclass
class TexturingCamSeq:
    cams: CamerasBase
    meshes: Meshes
    verts_uvs: torch.Tensor
    faces_uvs: torch.Tensor
    depth_maps: List[Image.Image]
    projections: List[TexelProjection]
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
        do_text_and_texture_resampling=True,
        use_update_masks=True,
        use_referecnce_kvs=True,
    ):
        self.pipe = pipe
        self.guidance_scale = guidance_scale
        self.uv_res = uv_res
        self.controlnet_conditioning_scale = controlnet_conditioning_scale
        self.do_text_and_texture_resampling = do_text_and_texture_resampling
        self.use_update_masks = use_update_masks
        self.use_referecnce_kvs = use_referecnce_kvs

    def set_attn_processor(self):
        module_paths = find_attn_modules(
            self.pipe.unet,
            block_types=[BlockType.UP],
            layer_types=[AttnType.SELF_ATTN],
            return_as_string=True,
        )

        # create attn processor
        self.attn_processor = FinalAttnProcessor(
            self.pipe.unet,
            do_kv_extraction=True,
            also_attend_to_self=True,
            attend_to_injected=True,
            kv_extraction_paths=module_paths,
        )

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
            extracted_kvs_uncond=extracted_kvs_uncond,
            extracted_kvs_cond=extracted_kvs_cond,
        )

    def compute_newly_visible_masks(self, seq: AnimSequence):
        raster_res = 1000
        projections = compute_texel_projections(
            seq.meshes,
            seq.cams,
            seq.verts_uvs,
            seq.faces_uvs,
            self.uv_res,
            raster_res=raster_res,
        )

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
            newly_visible_masks, (64, 64), thresh=0.2
        )

        return newly_visible_masks

    def compute_uv_update_masks(self, seq: AnimSequence, logger=NULL_LOGGER):
        raster_res = 1000
        projections = compute_texel_projections(
            seq.meshes,
            seq.cams,
            seq.verts_uvs,
            seq.faces_uvs,
            self.uv_res,
            raster_res=raster_res,
        )

        # compute quality maps based on image-space gradients
        quality_maps = [
            compute_uv_jacobian_map(c, m, seq.verts_uvs, seq.faces_uvs)
            for c, m in zip(seq.cams, seq.meshes)
        ]
        quality_maps = torch.stack(quality_maps)

        for i, q in enumerate(quality_maps):
            logger.write("quality_map", q, frame_i=i)

        # give some extra weight to first view
        quality_maps[0] /= 2

        # compute update masks
        better_quality_masks = compute_autoregressive_update_masks(
            seq.cams,
            seq.meshes,
            projections,
            quality_maps,
            self.uv_res,
            seq.verts_uvs,
            seq.faces_uvs,
            quality_factor=1.5,
        )

        # quality_factor 0: always update
        # quality factor 1.1: update if better quality
        # quality_factor high = newly visible
        return better_quality_masks

    def precompute_cam_seq(self, seq: AnimSequence, logger=NULL_LOGGER):
        # newly visible masks
        newly_visible_masks = self.compute_newly_visible_masks(seq)

        # projections and fragments
        PROJ_RASTER_RES = 1000
        projections = []
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
            projections.append(projection)

            # rasterize
            frags = rasterizer(mesh, cameras=seq.cams[i])
            fragments.append(frags)

        # depth maps
        depth_maps = seq.render_depth_maps()

        # update masks
        update_masks = self.compute_uv_update_masks(seq, logger=logger)

        return TexturingCamSeq(
            cams=seq.cams,
            meshes=seq.meshes,
            verts_uvs=seq.verts_uvs,
            faces_uvs=seq.faces_uvs,
            depth_maps=depth_maps,
            projections=projections,
            fragments=fragments,
            newly_visible_masks=newly_visible_masks,
            update_masks=update_masks,
        )

    def attn_guided_mv_sampling(
        self,
        latents,
        cond_embeddings,
        uncond_embeddings,
        t,
        cam_seq: TexturingCamSeq,
        prev_clean_tex: Optional[Tensor] = None,
        logger=NULL_LOGGER,
        generator=None,
    ):
        if prev_clean_tex is None:
            clean_tex = torch.zeros(self.uv_res, self.uv_res, 3).cuda()
        else:
            clean_tex = prev_clean_tex.clone()

        updated_latents = latents.clone()

        kvs_cond = {}
        kvs_uncond = {}

        for i in range(len(cam_seq.cams)):
            proj = cam_seq.projections[i]
            depth = cam_seq.depth_maps[i]
            latent = updated_latents[i]

            out = self.model_forward_guided_kvs(
                latent.unsqueeze(0),
                cond_embeddings[[i]],
                uncond_embeddings[[i]],
                t,
                depth_maps=[depth],
                injected_kvs_cond=kvs_cond,
                injected_kvs_uncond=kvs_uncond,
                extract_kvs=True,
            )

            # update ref-kvs
            if i == 0 and self.use_referecnce_kvs:
                kvs_cond = out.extracted_kvs_cond
                kvs_uncond = out.extracted_kvs_uncond

            # get clean image
            clean_im = self.pipe.scheduler.step(
                out.noise_pred, t, latent
            ).pred_original_sample
            clean_im_rgb = self.pipe.decode_latents(
                clean_im, output_type="pt", generator=generator
            )[0]

            logger.write("clean_im", clean_im_rgb, t=t, frame_i=i)

            # project texture
            if self.use_update_masks:
                update_mask = cam_seq.update_masks[i]
                project_view_to_texture_masked(
                    clean_tex, clean_im_rgb, update_mask, proj
                )
            else:
                update_uv_texture(clean_tex, clean_im_rgb, proj)

            # update next latent
            if i < len(cam_seq.cams) - 1:
                i_next = i + 1
                mesh_next = cam_seq.meshes[i_next]
                cam_next = cam_seq.cams[i_next]
                latent_next = updated_latents[i_next]
                mask_i_next = cam_seq.newly_visible_masks[i_next].cuda()

                # render partial texture
                rendered = render_texture(
                    mesh_next, cam_next, clean_tex, cam_seq.verts_uvs, cam_seq.faces_uvs
                )[0]
                logger.write("rendered", rendered, t=t, frame_i=i_next)

                # bring render to noise level
                rendered_latent = self.pipe.encode_images([rendered], generator)[0]
                noise = torch.randn_like(rendered_latent)
                rendered_noisy = self.pipe.scheduler.add_noise(
                    rendered_latent, noise, t
                )

                updated_latent = latent_next * mask_i_next + rendered_noisy * (
                    1 - mask_i_next
                )

                # blend according to mask
                updated_latents[i_next] = updated_latent

        return updated_latents, clean_tex

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

    def txt_and_texture_guided_resampling(
        self, clean_tex, latents, t, cond_embeddings, uncond_embeddings, cam_seq
    ):
        rendered_noise = self.render_noise_preds(
            clean_tex,
            latents,
            t,
            cam_seq,
        )

        if not self.do_text_and_texture_resampling:
            return rendered_noise

        self.attn_processor.injected_kvs = {}
        latents_duplicated = torch.cat([latents] * 2)
        both_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        depth_maps_duplicated = cam_seq.depth_maps * 2
        noise_preds = self.pipe.model_forward(
            latents_duplicated, both_embeddings, t, depth_maps_duplicated
        )
        noise_pred_uncond, noise_pred_cond = noise_preds.chunk(2)

        w = self.guidance_scale
        noise_tex = (1 / w) * (rendered_noise - noise_pred_uncond) + noise_pred_uncond

        progress = self.pipe.denoising_progress(t)
        w_texture = w * progress
        w_text = w * (1 - progress)

        noise_pred = (
            noise_pred_uncond
            + w_text * (noise_pred_cond - noise_pred_uncond)
            + w_texture * (noise_tex - noise_pred_uncond)
        )

        return noise_pred


@dataclass
class TexturingConfig:
    num_inference_steps: int = 15
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    uv_res: int = 600
    do_text_and_texture_resampling: bool = True
    use_update_masks: bool = True
    use_referecnce_kvs: bool = True
    use_prev_clean_tex: bool = True


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
            do_text_and_texture_resampling=conf.do_text_and_texture_resampling,
            use_update_masks=conf.use_update_masks,
            use_referecnce_kvs=conf.use_referecnce_kvs,
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

        clean_tex = torch.zeros(method.uv_res, method.uv_res, 3).cuda()

        # denoising loop
        for t in tqdm(timesteps):
            # Stage 1: autoregressively denoise to get denoised texture pred

            if not (conf.use_update_masks and conf.use_prev_clean_tex):
                clean_tex = torch.zeros_like(clean_tex)

            latents, clean_tex = method.attn_guided_mv_sampling(
                latents,
                cond_embeddings,
                uncond_embeddings,
                t,
                cam_seq,
                logger=logger,
                generator=generator,
                prev_clean_tex=clean_tex,
            )

            logger.write(
                "clean_tex",
                clean_tex,
                t=t,
            )

            # Stage 2: obtain noise predictions
            noise_pred = method.txt_and_texture_guided_resampling(
                clean_tex,
                latents,
                t,
                cond_embeddings,
                uncond_embeddings,
                cam_seq,
            )

            # denoise latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        return self.decode_latents(latents, generator=generator)
