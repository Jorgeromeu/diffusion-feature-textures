from typing import Dict, List

import torch
import torchvision.transforms.functional as TF
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
    update_uv_texture,
)
from text3d2video.pipelines.controlnet_pipeline import BaseControlNetPipeline
from text3d2video.rendering import (
    compute_newly_visible_masks,
    downsample_masks,
    make_mesh_rasterizer,
    render_depth_map,
    render_texture,
    shade_mesh,
)
from text3d2video.util import chunk_dim
from text3d2video.utilities.logging import GrLogger, H5Logger


@dataclass
class TexGenConfig:
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    module_paths: List[str]


@dataclass
class TexGenModelOutput:
    noise_pred: Tensor
    extracted_kvs_uncond: Dict[str, Float[Tensor, "b t d"]]
    extracted_kvs_cond: Dict[str, Float[Tensor, "b t d"]]


class TexGenPipeline(BaseControlNetPipeline):
    conf: TexGenConfig
    attn_processor: FinalAttnProcessor
    logger: GrLogger

    def model_forward(
        self,
        latents: Float[Tensor, "b c h w"],
        embeddings: Float[Tensor, "b t d"],
        t: int,
        depth_maps: List[Image.Image],
    ) -> Tensor:
        # ControlNet Pass
        processed_ctrl_images = self.preprocess_controlnet_images(depth_maps)
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            latents,
            t,
            encoder_hidden_states=embeddings,
            controlnet_cond=processed_ctrl_images,
            conditioning_scale=self.conf.controlnet_conditioning_scale,
            guess_mode=False,
            return_dict=False,
        )

        # UNet Pass
        noise_pred = self.unet(
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
        noise_pred = noise_pred_uncond + self.conf.guidance_scale * (
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

        # setup configs
        self.conf = texgen_config

        # configure scheduler
        self.scheduler.set_timesteps(self.conf.num_inference_steps)

        # setup logger
        if logger is not None:
            self.logger = logger
        else:
            self.logger = H5Logger.create_disabled()

            # set up attn processor

        self.attn_processor = FinalAttnProcessor(
            self.unet,
            do_kv_extraction=True,
            also_attend_to_self=True,
            attend_to_injected=True,
            kv_extraction_paths=self.conf.module_paths,
        )
        self.unet.set_attn_processor(self.attn_processor)

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
        uv_res = 600
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
                raster_res=image_res,
                texture_res=uv_res,
            )
            projections.append(projection)

            # rasterize
            frags = rasterizer(meshes[i], cameras=cameras[i])
            fragments.append(frags)

        # precompute newly-visible masks
        newyly_visible_masks = compute_newly_visible_masks(
            cameras, meshes, projections, uv_res, image_res, verts_uvs, faces_uvs
        )
        newly_visible_masks_down = downsample_masks(
            newyly_visible_masks, (64, 64), thresh=0.1
        ).cuda()

        # denoising loop
        for t in tqdm(self.scheduler.timesteps):
            # Stage 1: autoregressively denoise to get RGB texture of denoised observation at t
            ref_kvs_cond = None
            ref_kvs_uncond = None
            clean_tex = torch.zeros(uv_res, uv_res, 3).cuda()
            for i in range(n_frames):
                mesh = meshes[i]
                cam = cameras[i]
                projection = projections[i]
                frags = fragments[i]
                depth_map = depth_maps[i]

                # newly visible pixels in view i
                mask_i = newly_visible_masks_down[i]

                # render clean image texture
                rendered_clean = shade_mesh(
                    mesh, frags, clean_tex, verts_uvs, faces_uvs
                )

                logger.write("rendered_clean", rendered_clean, frame_i=i, t=t)
                rendered_latent = self.encode_images([rendered_clean], generator)[0]

                # bring render to noise level
                epsilon = torch.randn_like(rendered_latent)
                rendered_noisy = self.scheduler.add_noise(rendered_latent, epsilon, t)

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

                denoised_observation = self.scheduler.step(
                    noise_pred_render,
                    t,
                    blended_latent,
                ).pred_original_sample

                # update clean image texture
                denoised_observation_rgb = self.decode_latents(
                    denoised_observation.unsqueeze(0), output_type="pt"
                )[0]

                logger.write("denoised_rgb", denoised_observation_rgb, frame_i=i, t=t)

                # update texture
                # TODO replace with system that updates based on image space coordinates
                update_uv_texture(
                    clean_tex,
                    denoised_observation_rgb,
                    projection.xys,
                    projection.uvs,
                    update_empty_only=True,
                )

                logger.write("clean_tex", clean_tex, t=t, frame_i=i)

            # Stage 2: obtain noise predictions
            noise_preds = []
            for i in range(n_frames):
                mesh = meshes[i]
                cam = cameras[i]

                # render view-consistent denoised observation for view i
                rendered_clean = render_texture(
                    mesh, cam, clean_tex, verts_uvs, faces_uvs
                )[0]
                rendered_latent = self.encode_images([rendered_clean], generator)[0]

                # convert denoised observation to noise prediction
                alpha_cumprod_t = self.scheduler.alphas_cumprod[t]
                sqrt_alpha = alpha_cumprod_t.sqrt()
                sqrt_one_minus_alpha = (1.0 - alpha_cumprod_t).sqrt()

                noisy = latents[i]
                noise_pred_render = (
                    noisy - sqrt_alpha * rendered_latent
                ) / sqrt_one_minus_alpha

                noise_preds.append(noise_pred_render)

            noise_preds = torch.stack(noise_preds, dim=0)

            # denoise latents
            latents = self.scheduler.step(noise_preds, t, latents).prev_sample
            self.logger.flush()

        for i, latent in enumerate(latents):
            self.logger.write("latent", latent, frame_i=i, t=0)

        # decode latents in chunks
        decoded_imgs = []
        chunks_indices = torch.split(torch.arange(0, n_frames), 5)
        for chunk_frame_indices in chunks_indices:
            chunk_latents = latents[chunk_frame_indices]
            chunk_images = self.decode_latents(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        return decoded_imgs
