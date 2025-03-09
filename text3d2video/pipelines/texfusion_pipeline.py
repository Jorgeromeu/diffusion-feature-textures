from typing import List

import torch
from attr import dataclass
from jaxtyping import Float
from PIL.Image import Image
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    MeshRenderer,
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
    TextureShader,
    make_mesh_rasterizer,
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

        uv_res = 64
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

        # initialize latents from standard normal
        uv_res = 64
        latent_texture = torch.randn(
            uv_res, uv_res, 4, generator=generator, dtype=self.dtype, device=self.device
        )

        rasterizer = make_mesh_rasterizer(resolution=64)
        shader = TextureShader()
        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

        # render latent texture
        meshes.textures = make_repeated_uv_texture(
            latent_texture, faces_uvs, verts_uvs, sampling_mode="nearest", N=n_views
        )

        for t in tqdm(self.scheduler.timesteps):
            latents = renderer(meshes, cameras=cameras).to(latent_texture)

            noise_pred = self.model_forward_guided(
                latents,
                t,
                cond_embeddings,
                uncond_embeddings,
                depth_maps,
            )

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            update_uv_texture(
                latent_texture,
                latents[0],
                texel_xys[0],
                texel_uvs[0],
                interpolation="nearest",
            )

        return self.decode_latents(latents)

        # denoising loop
        for t in tqdm(self.scheduler.timesteps):
            denoised_views = []
            for i in range(0, n_views):
                view_cam = cameras[i]
                view_mesh = meshes[i]

                # render latent texture
                view_mesh.textures = make_repeated_uv_texture(
                    latent_texture, faces_uvs, verts_uvs, sampling_mode="nearest"
                )
                render = renderer(view_mesh, cameras=view_cam).to(latent_texture)

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

                # update latent texture
                update_uv_texture(
                    latent_texture,
                    denoised,
                    texel_xys[i],
                    texel_uvs[i],
                    interpolation="nearest",
                )

            latents = torch.stack(denoised_views)

        return self.decode_latents(latents)
