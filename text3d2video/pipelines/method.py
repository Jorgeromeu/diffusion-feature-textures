from typing import List

import torch
from jaxtyping import Float
from PIL import Image
from rerun import Tensor

from text3d2video.backprojection import update_uv_texture
from text3d2video.noise_initialization import UVNoiseInitializer
from text3d2video.pipelines.controlnet_pipeline import BaseControlNetPipeline
from text3d2video.rendering import render_texture


class MethodLogic:
    def __init__(self, pipe: BaseControlNetPipeline, guidance_scale=7.5):
        self.pipe = pipe
        self.guidance_scale = guidance_scale

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

    def denoise_multiview(
        self,
        meshes,
        cameras,
        verts_uvs,
        faces_uvs,
        projections,
        fragments,
        depths,
        newly_visible_masks_down,
        latents,
        cond_embeddings,
        uncond_embeddings,
        t,
    ):
        uv_res = 600
        clean_tex = torch.zeros(uv_res, uv_res, 3).cuda()

        uv_noise = UVNoiseInitializer()
        uv_noise.sample_background()
        uv_noise.sample_noise_texture()

        for i in range(len(meshes)):
            mesh = meshes[i]
            cam = cameras[i]
            proj = projections[i]
            frags = fragments[i]
            depth = depths[i]
            latent = latents[i]
            newly_visible = newly_visible_masks_down[i].cuda()

            # render texture
            rendered = render_texture(mesh, cam, clean_tex, verts_uvs, faces_uvs)

            # bring render to noise level
            rendered_latent = self.encode_images(rendered)
            noise = uv_noise.initial_noise(mesh, cam, verts_uvs, faces_uvs)
            noise = torch.randn_like(noise)
            rendered_noisy = self.scheduler.add_noise(rendered_latent, noise, t)[0]

            blended_latent = latent * newly_visible + rendered_noisy * (
                1 - newly_visible
            )
            blended_latent = blended_latent.to(latent)

            # blended_latent = latent
            blended_latent = blended_latent.unsqueeze(0)

            # noise pred
            noise_pred = self.model_forward_guided(
                blended_latent,
                cond_embeddings[[i]],
                uncond_embeddings[[i]],
                t,
                depth_maps=[depth],
            )

            # get clean image
            clean_im = self.pipe.scheduler.step(
                noise_pred, t, blended_latent
            ).pred_original_sample
            clean_im_rgb = self.pipe.decode_latents(clean_im, output_type="pt")[0]

            update_uv_texture(clean_tex, clean_im_rgb, proj.xys, proj.uvs)
