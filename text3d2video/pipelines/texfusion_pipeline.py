from typing import List

import torch
from attr import dataclass
from jaxtyping import Float
from PIL.Image import Image
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
)
from pytorch3d.structures import Meshes
from torch import Tensor
from tqdm import tqdm

from text3d2video.backprojection import (
    project_visible_texels_to_camera,
    update_uv_texture,
)
from text3d2video.pipelines.controlnet_pipeline import BaseControlNetPipeline
from text3d2video.rendering import (
    make_mesh_renderer,
    make_repeated_uv_texture,
    render_depth_map,
)


# pylint: disable=too-many-instance-attributes
@dataclass
class TexturingConfig:
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    module_paths: list[str]


class TexFusionPipeline(BaseControlNetPipeline):
    conf: TexturingConfig

    def model_forward_guided(
        self,
        latents: Float[Tensor, "b c h w"],
        t: int,
        cond_embeddings: Float[Tensor, "b t c"],
        uncond_embeddings: Float[Tensor, "b t c"],
        depth_maps: List[Image],
    ) -> Float[Tensor, "b c h w"]:
        # perform cond and uncond passes through model
        latents_both = torch.cat([latents] * 2, dim=0)
        embeddings_both = torch.cat([cond_embeddings, uncond_embeddings])
        depth_maps_both = depth_maps * 2
        noises = self.model_forward(
            latents_both,
            t,
            embeddings_both,
            depth_maps_both,
        )

        noise_cond, noise_uncond = noises.chunk(2)

        noise = noise_uncond + 7.5 * (noise_cond - noise_uncond)

        return noise

    def model_forward(
        self,
        latents: Float[Tensor, "b c h w"],
        t: int,
        embeddings: Float[Tensor, "b t c"],
        depth_maps: List[Image],
    ) -> Float[Tensor, "b c h w"]:
        # ControlNet pass
        processed_ctrl_images = self.preprocess_controlnet_images(depth_maps)
        down_residuals, mid_residual = self.controlnet(
            latents,
            t,
            encoder_hidden_states=embeddings,
            controlnet_cond=processed_ctrl_images,
            conditioning_scale=self.conf.controlnet_conditioning_scale,
            guess_mode=False,
            return_dict=False,
        )

        # UNet pass with residuals
        noise_pred = self.unet(
            latents,
            t,
            mid_block_additional_residual=mid_residual,
            down_block_additional_residuals=down_residuals,
            encoder_hidden_states=embeddings,
        ).sample

        return noise_pred

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        meshes: Meshes,
        cameras: FoVPerspectiveCameras,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        conf: TexturingConfig,
        generator=None,
    ):
        self.conf = conf

        # number of images being generated
        n_views = len(meshes)

        # render depth maps
        depth_maps = render_depth_map(meshes, cameras, 512)

        # encode prompt
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt] * n_views)

        uv_res = 120
        texel_xys = []
        texel_uvs = []
        for cam, view_mesh in zip(cameras, meshes):
            xys, uvs = project_visible_texels_to_camera(
                view_mesh, cam, verts_uvs, faces_uvs, uv_res, raster_res=2000
            )
            texel_xys.append(xys)
            texel_uvs.append(uvs)

        # set timesteps
        self.scheduler.set_timesteps(conf.num_inference_steps)

        latent_tex = torch.randn(
            uv_res, uv_res, 4, generator=generator, dtype=self.dtype, device=self.device
        )
        renderer = make_mesh_renderer(64)

        # Attempt 1: normal denoising
        # latents = torch.randn(
        #     n_views,
        #     4,
        #     64,
        #     64,
        #     generator=generator,
        #     dtype=self.dtype,
        #     device=self.device,
        # )

        # for t in tqdm(self.scheduler.timesteps):
        #     noise_pred = self.model_forward_guided(
        #         latents,
        #         t,
        #         cond_embeddings,
        #         uncond_embeddings,
        #         depth_maps,
        #     )

        #     latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        # return self.decode_latents(latents)

        # Attempt 2: denoise single view in uv space
        # latent_tex = torch.randn(
        #     uv_res, uv_res, 4, generator=generator, dtype=self.dtype, device=self.device
        # )

        # cam = cameras[0]
        # mesh = meshes[0]

        # for t in tqdm(self.scheduler.timesteps):
        #     render_meshes = mesh.clone()
        #     render_meshes.textures = make_repeated_uv_texture(
        #         latent_tex,
        #         faces_uvs,
        #         verts_uvs,
        #         sampling_mode="nearest",
        #         N=len(render_meshes),
        #     )
        #     rendered = renderer(render_meshes, cameras=cam)
        #     rendered = rendered.to(latent_tex)

        #     noise_pred = self.model_forward_guided(
        #         rendered,
        #         t,
        #         cond_embeddings[[0]],
        #         uncond_embeddings[[0]],
        #         [depth_maps[0]],
        #     )

        #     denoised = self.scheduler.step(
        #         noise_pred, t, rendered, generator=generator
        #     ).prev_sample

        #     update_uv_texture(
        #         latent_tex,
        #         denoised[0],
        #         texel_xys[0],
        #         texel_uvs[0],
        #         update_empty_only=False,
        #         interpolation="nearest",
        #     )

        # return self.decode_latents(denoised)

        # Attempt 3: SIMS
        for t in tqdm(self.scheduler.timesteps):
            denoised_views = []

            mask = torch.zeros(uv_res, uv_res, 1, device=self.device)

            for i in range(0, n_views):
                view_cam = cameras[i]
                view_mesh = meshes[i]

                # render latent texture
                view_mesh.textures = make_repeated_uv_texture(
                    latent_tex, faces_uvs, verts_uvs, sampling_mode="nearest"
                )
                render = renderer(view_mesh, cameras=view_cam).to(latent_tex)

                # TODO
                # add appropriate noise according to mask

                # denoise view
                noise_pred = self.model_forward_guided(
                    render,
                    t,
                    cond_embeddings[[i]],
                    uncond_embeddings[[i]],
                    [depth_maps[i]],
                )
                denoised = self.scheduler.step(noise_pred, t, render).prev_sample[0]
                denoised_views.append(denoised)

                # update mask
                update_uv_texture(
                    mask,
                    torch.ones(1, uv_res, uv_res, device=self.device),
                    texel_xys[i],
                    texel_uvs[i],
                    interpolation="nearest",
                    update_empty_only=False,
                )

                # update latent texture
                update_uv_texture(
                    latent_tex,
                    denoised,
                    texel_xys[i],
                    texel_uvs[i],
                    interpolation="nearest",
                    update_empty_only=False,
                )

            latents = torch.stack(denoised_views)

        return self.decode_latents(latents)
