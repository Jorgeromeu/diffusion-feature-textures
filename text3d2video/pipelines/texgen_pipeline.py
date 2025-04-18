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
from text3d2video.utilities.logging import H5Logger


@dataclass
class TexGenConfig:
    module_paths: List[str]
    num_inference_steps: int = 10
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    quality_update_factor: float = 1.5
    uv_res: int = 600


@dataclass
class TexGenModelOutput:
    noise_pred: Tensor
    extracted_kvs_uncond: Dict[str, Float[Tensor, "b t d"]]
    extracted_kvs_cond: Dict[str, Float[Tensor, "b t d"]]


class TexGenLogic:
    """
    Implement the behavior of the TexGen Pipeline
    """

    def __init__(
        self,
        pipe: BaseControlNetPipeline,
        uv_res: int,
        guidance_scale: float,
        controlnet_scale: float,
        module_paths: List[str],
    ):
        self.pipe = pipe
        self.uv_res = uv_res
        self.guidance_scale = guidance_scale
        self.controlnet_scale = controlnet_scale

        # create attn processor
        self.attn_processor = FinalAttnProcessor(
            self.pipe.unet,
            do_kv_extraction=True,
            also_attend_to_self=True,
            attend_to_injected=True,
            kv_extraction_paths=module_paths,
        )

    def set_attn_processor(self):
        self.pipe.unet.set_attn_processor(self.attn_processor)

    def model_forward(
        self,
        latents: Float[Tensor, "b c h w"],
        embeddings: Float[Tensor, "b t d"],
        t: int,
        depth_maps: List[Image.Image],
    ) -> Tensor:
        # ControlNet Pass
        processed_ctrl_images = self.pipe.preprocess_controlnet_images(depth_maps)
        down_block_res_samples, mid_block_res_sample = self.pipe.controlnet(
            latents,
            t,
            encoder_hidden_states=embeddings,
            controlnet_cond=processed_ctrl_images,
            conditioning_scale=self.controlnet_scale,
            guess_mode=False,
            return_dict=False,
        )

        # UNet Pass
        noise_pred = self.pipe.unet(
            latents,
            t,
            mid_block_additional_residual=mid_block_res_sample,
            down_block_additional_residuals=down_block_res_samples,
            encoder_hidden_states=embeddings,
        ).sample

        return noise_pred

    def model_forward_guided(
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
        noise_pred = self.model_forward(
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

    def predict_clean_texture(
        self,
        latents,
        cond_embeddings,
        uncond_embeddings,
        t,
        meshes,
        verts_uvs,
        faces_uvs,
        projections,
        fragments,
        depth_maps,
        newly_visible_masks_down,
        better_quality_masks,
        generator=None,
    ):
        ref_kvs_cond = None
        ref_kvs_uncond = None
        # initialize clean image texture
        clean_tex = torch.zeros(self.uv_res, self.uv_res, 3).cuda()

        # iterate over views
        for i in range(len(meshes)):
            mesh = meshes[i]
            projection = projections[i]
            frags = fragments[i]
            depth_map = depth_maps[i]

            # newly visible pixels in view i
            mask_i = newly_visible_masks_down[i]

            # render clean image partial texture
            rendered_clean = shade_mesh(mesh, frags, clean_tex, verts_uvs, faces_uvs)
            # self.logger.write("rendered_clean", rendered_clean, frame_i=i, t=t)
            # encode rendered clean image
            rendered_latent = self.pipe.encode_images([rendered_clean], generator)[0]

            # bring render to noise level
            epsilon = torch.randn_like(rendered_latent)
            rendered_noisy = self.pipe.scheduler.add_noise(rendered_latent, epsilon, t)

            # blend latent with rendered noisy
            latent = latents[i]
            blended_latent = latent * mask_i + rendered_noisy * (1 - mask_i)
            blended_latent = blended_latent.to(latent)

            # predict denoised observation
            model_out = self.model_forward_guided(
                blended_latent.unsqueeze(0),
                cond_embeddings[[i]],
                uncond_embeddings[[i]],
                t,
                [depth_map],
                extract_kvs=True,
                injected_kvs_cond=ref_kvs_cond,
                injected_kvs_uncond=ref_kvs_uncond,
            )
            noise_pred_render = model_out.noise_pred[0]

            if i == 0:
                ref_kvs_cond = model_out.extracted_kvs_cond
                ref_kvs_uncond = model_out.extracted_kvs_uncond

            denoised_observation = self.pipe.scheduler.step(
                noise_pred_render,
                t,
                blended_latent,
            ).pred_original_sample

            denoised_observation_rgb = self.pipe.decode_latents(
                denoised_observation.unsqueeze(0), output_type="pt"
            )[0]
            # self.logger.write("denoised_rgb", denoised_observation_rgb, frame_i=i, t=t)

            # update texture based on quality maps
            update_mask = better_quality_masks[i]

            # filter out texels that lie in the outside the update mask
            texels_in_mask = sample_feature_map_ndc(
                update_mask.unsqueeze(0).cuda(), projection.xys
            )
            uvs = projection.uvs[texels_in_mask[:, 0]]
            xys = projection.xys[texels_in_mask[:, 0]]

            # update texture
            colors = sample_feature_map_ndc(denoised_observation_rgb, xys).float()
            clean_tex[uvs[:, 1], uvs[:, 0]] = colors

        return clean_tex

    def render_noise_preds(
        self,
        clean_tex,
        meshes,
        cameras,
        verts_uvs,
        faces_uvs,
        latents,
        t,
        generator=None,
    ):
        noise_preds = []
        for i in range(len(cameras)):
            mesh = meshes[i]
            cam = cameras[i]

            # render view-consistent denoised observation for view i
            rendered_clean = render_texture(mesh, cam, clean_tex, verts_uvs, faces_uvs)[
                0
            ]
            rendered_latent = self.pipe.encode_images([rendered_clean], generator)[0]

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


class TexGenPipeline(BaseControlNetPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        meshes: Meshes,
        cameras: FoVPerspectiveCameras,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        texgen_config: TexGenConfig,
        prompt_suffixes: List[str] = None,
        generator=None,
        logger=None,
    ):
        n_frames = len(meshes)

        # make texgen logic object
        texgen = TexGenLogic(
            self,
            texgen_config.uv_res,
            texgen_config.guidance_scale,
            texgen_config.controlnet_conditioning_scale,
            texgen_config.module_paths,
        )

        # configure scheduler
        self.scheduler.set_timesteps(texgen_config.num_inference_steps)

        # setup logger
        if logger is None:
            logger = H5Logger.create_disabled()

        # augment prompts
        prompts = [prompt] * n_frames
        if prompt_suffixes is not None:
            prompts = [p + s for p, s in zip(prompts, prompt_suffixes)]

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt(prompts)

        # render depth maps for frames
        depth_maps = render_depth_map(meshes, cameras, 512)

        # initial latent noise
        latents = self.prepare_latents(len(meshes), generator=generator)

        # TODO move constants
        image_res = 512

        # precompute rasterization and projections
        projections = []
        fragments = []
        rasterizer = make_mesh_rasterizer(resolution=image_res)
        for i in range(n_frames):
            cam = cameras[i]
            mesh = meshes[i]

            # project UVs to camera
            projection = project_visible_texels_to_camera(
                mesh,
                cam,
                verts_uvs,
                faces_uvs,
                texture_res=texgen_config.uv_res,
                raster_res=2000,
            )
            projections.append(projection)

            # rasterize
            frags = rasterizer(meshes[i], cameras=cameras[i])
            fragments.append(frags)

        # precompute newly-visible masks
        newly_visible_masks = compute_newly_visible_masks(
            cameras,
            meshes,
            projections,
            texgen_config.uv_res,
            image_res,
            verts_uvs,
            faces_uvs,
        )
        newly_visible_masks_down = downsample_masks(
            newly_visible_masks, (64, 64), thresh=0.1
        ).cuda()

        # compute quality maps
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
            texgen_config.uv_res,
            verts_uvs,
            faces_uvs,
            quality_factor=texgen_config.quality_update_factor,
        )

        texgen.set_attn_processor()

        # denoising loop
        for t in tqdm(self.scheduler.timesteps):
            # Stage 1: autoregressively denoise to get RGB texture of denoised observation at t
            clean_tex = texgen.predict_clean_texture(
                latents,
                cond_embeddings,
                uncond_embeddings,
                t,
                meshes,
                verts_uvs,
                faces_uvs,
                projections,
                fragments,
                depth_maps,
                newly_visible_masks_down,
                better_quality_masks,
                generator=generator,
            )

            # Stage 2: obtain noise predictions
            noise_preds = texgen.render_noise_preds(
                clean_tex,
                meshes,
                cameras,
                verts_uvs,
                faces_uvs,
                latents,
                t,
                generator=generator,
            )

            # denoise latents
            latents = self.scheduler.step(noise_preds, t, latents).prev_sample

        # decode latents in chunks
        decoded_imgs = []
        chunks_indices = torch.split(torch.arange(0, n_frames), 5)
        for chunk_frame_indices in chunks_indices:
            chunk_latents = latents[chunk_frame_indices]
            chunk_images = self.decode_latents(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        return decoded_imgs
