import torch
import torchvision.transforms.functional as TF
from attr import dataclass
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
)
from pytorch3d.structures import Meshes
from tqdm import tqdm

from text3d2video.backprojection import (
    project_visible_texels_to_camera,
)
from text3d2video.pipelines.controlnet_pipeline import BaseControlNetPipeline
from text3d2video.rendering import render_depth_map
from text3d2video.util import sample_feature_map_ndc


@dataclass
class TexGenConfig:
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float


class TexGenPipeline(BaseControlNetPipeline):
    texgen_config: TexGenConfig

    def model_fwd_cfg(self, latents, depths, t, cond_embeddings, uncond_embeddings):
        latents_duplicated = torch.cat([latents] * 2)
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        # # ControlNet Pass
        processed_ctrl_images = self.preprocess_controlnet_images(depths)
        processed_ctrl_images = torch.cat([processed_ctrl_images] * 2)
        down_residuals, mid_residual = self.controlnet(
            latents_duplicated,
            t,
            encoder_hidden_states=text_embeddings,
            controlnet_cond=processed_ctrl_images,
            conditioning_scale=self.texgen_config.controlnet_conditioning_scale,
            guess_mode=False,
            return_dict=False,
        )

        noise_preds = self.unet(
            latents_duplicated,
            t,
            encoder_hidden_states=text_embeddings,
            mid_block_additional_residual=mid_residual,
            down_block_additional_residuals=down_residuals,
        ).sample

        noise_preds_uncond, noise_preds_cond = noise_preds.chunk(2)
        noise_preds_cfg = noise_preds_uncond + self.texgen_config.guidance_scale * (
            noise_preds_cond - noise_preds_uncond
        )

        return noise_preds_cfg

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        meshes: Meshes,
        cameras: FoVPerspectiveCameras,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        texgen_config: TexGenConfig,
        generator=None,
    ):
        self.texgen_config = texgen_config

        n_views = len(meshes)

        # precompute visible-vert rasterization for each frame
        view_texel_xys = []
        view_texel_uvs = []
        for cam, mesh in zip(cameras, meshes):
            xys, uvs = project_visible_texels_to_camera(
                mesh, cam, verts_uvs, faces_uvs, 512
            )
            view_texel_xys.append(xys)
            view_texel_uvs.append(uvs)

        # render depths
        depth_maps = render_depth_map(meshes, cameras, 512)

        # Get prompt embeddings for guidance
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt] * n_views)

        # set timesteps
        self.scheduler.set_timesteps(texgen_config.num_inference_steps)

        # sample latents
        latents = torch.randn(
            n_views,
            4,
            64,
            64,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

        for _, t in enumerate(tqdm(self.scheduler.timesteps)):
            # 1.Attnetion Guided Multi-View Sampling

            all_prev_samples = []

            texture = torch.zeros(512, 512, 3).to(self.device)

            for i in range(n_views):
                # get depth, latent and embeddings
                view_depth = [depth_maps[i]]
                view_latent = latents[i].unsqueeze(0)
                view_cond_embeddings = cond_embeddings[i].unsqueeze(0)
                view_uncond_embeddings = uncond_embeddings[i].unsqueeze(0)

                # get noise pred
                view_noise_pred = self.model_fwd_cfg(
                    view_latent,
                    view_depth,
                    t,
                    view_cond_embeddings,
                    view_uncond_embeddings,
                )[0]

                # get clean image
                view_clean_latent = self.scheduler.step(
                    view_noise_pred, t, view_latent
                ).pred_original_sample[0]

                # decode clean image
                decoded = self.decode_latents(view_clean_latent.unsqueeze(0))[0]
                decoded_pt = TF.to_tensor(decoded)

                # update texture
                xys = view_texel_xys[i]
                uvs = view_texel_uvs[i]
                texel_colors = sample_feature_map_ndc(decoded_pt, xys.cpu())
                texture[uvs[:, 1], uvs[:, 0]] = texel_colors.cuda()

                # get prev sample
                view_prev_sample = self.scheduler.step(
                    view_noise_pred, t, view_latent
                ).prev_sample[0]

                all_prev_samples.append(view_prev_sample)

            latents = torch.stack(all_prev_samples)

        return self.decode_latents(latents)
