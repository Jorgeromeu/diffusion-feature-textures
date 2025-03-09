from typing import Dict, List

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
from torchmetrics import Dice
from tqdm import tqdm

from text3d2video.attn_processors.extraction_injection_attn import (
    UnifiedAttnProcessor,
)
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
from text3d2video.util import map_dict


# pylint: disable=too-many-instance-attributes
@dataclass
class TexturingConfig:
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    module_paths: list[str]


def classifier_free_guidance(noise_pred_cond, noise_pred_uncond, guidance_scale):
    return noise_pred_cond + guidance_scale * (noise_pred_cond - noise_pred_uncond)


@dataclass
class TexturingModelOutput:
    noise_preds: Float[Tensor, "b c h w"]
    extracted_kvs: Dict[str, Float[Tensor, "b t c"]]
    extracted_qrys: Dict[str, Float[Tensor, "b c h w"]]


@dataclass
class TexturingGuidedModelOutput:
    noise_preds: Float[Tensor, "b c h w"]
    extracted_kvs_cond: Dict[str, Float[Tensor, "b t c"]]
    extracted_qrys_cond: Dict[str, Float[Tensor, "b t c"]]
    extracted_kvs_uncond: Dict[str, Float[Tensor, "b t c"]]
    extracted_qrys_uncond: Dict[str, Float[Tensor, "b c h w"]]


class TexturingPipeline(BaseControlNetPipeline):
    attn_processor: UnifiedAttnProcessor
    conf: TexturingConfig

    def model_forward_guided(
        self,
        latents: Float[Tensor, "b c h w"],
        t: int,
        cond_embeddings: Float[Tensor, "b t c"],
        uncond_embeddings: Float[Tensor, "b t c"],
        depth_maps: List[Image],
        injected_kvs_cond: Float[Tensor, "b t c"] = None,
        injected_kvs_uncond: Float[Tensor, "b t c"] = None,
        injected_qrys_cond: Float[Tensor, "b c h w"] = None,
        injected_qrys_uncond: Float[Tensor, "b c h w"] = None,
        extract_kvs: bool = False,
        extract_qrys: bool = False,
    ) -> TexturingGuidedModelOutput:
        # perform multiple passes through model and compose noise predictions

        out_cond = self.model_forward(
            latents,
            t,
            cond_embeddings,
            depth_maps,
            # injected features
            injected_qrys=injected_qrys_cond,
            injected_kvs=injected_kvs_cond,
            # extraction settings
            extract_kvs=extract_kvs,
            extract_qrys=extract_qrys,
        )

        out_uncond = self.model_forward(
            latents,
            t,
            uncond_embeddings,
            depth_maps,
            # injected features
            injected_qrys=injected_qrys_uncond,
            injected_kvs=injected_kvs_uncond,
            # extraction settings
            extract_kvs=extract_kvs,
            extract_qrys=extract_qrys,
        )

        noise_cond = out_cond.noise_preds
        noise_uncond = out_uncond.noise_preds

        noise = noise_uncond + 7.5 * (noise_cond - noise_uncond)

        return TexturingGuidedModelOutput(
            noise,
            out_cond.extracted_kvs,
            out_cond.extracted_qrys,
            out_uncond.extracted_kvs,
            out_uncond.extracted_qrys,
        )

    def model_forward(
        self,
        latents,
        t,
        embeddings,
        depth_maps,
        injected_kvs=None,
        injected_qrys=None,
        extract_kvs=False,
        extract_qrys=False,
    ):
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

        self.attn_processor.clear_injected_features()
        self.attn_processor.disable_all_extraction()
        self.attn_processor.do_kv_extraction = extract_kvs
        self.attn_processor.do_spatial_qry_extraction = extract_qrys

        # pass injected features
        if injected_kvs is not None:
            self.attn_processor.injected_kvs = injected_kvs

        if injected_qrys is not None:
            self.attn_processor.injected_qrys = injected_qrys

        # UNet pass with residuals
        noise_pred = self.unet(
            latents,
            t,
            mid_block_additional_residual=mid_residual,
            down_block_additional_residuals=down_residuals,
            encoder_hidden_states=embeddings,
        ).sample

        extracted_kvs = self.attn_processor.extracted_kvs
        extracted_qrys = self.attn_processor.extracted_spatial_qrys

        self.attn_processor.clear_extracted_features()

        return TexturingModelOutput(noise_pred, extracted_kvs, extracted_qrys)

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

        # setup attn processor
        self.attn_processor = UnifiedAttnProcessor(
            self.unet,
            also_attend_to_self=True,
            feature_blend_alhpa=1.0,
            kv_extraction_paths=conf.module_paths,
            spatial_qry_extraction_paths=conf.module_paths,
        )
        self.unet.set_attn_processor(self.attn_processor)

        # number of images being generated
        batch_size = len(meshes)

        # render depth maps
        depth_maps = render_depth_map(meshes, cameras, 512)

        uv_res = 100
        texel_xys = []
        texel_uvs = []
        for cam, view_mesh in zip(cameras, meshes):
            xys, uvs = project_visible_texels_to_camera(
                view_mesh, cam, verts_uvs, faces_uvs, uv_res, raster_res=2000
            )
            texel_xys.append(xys)
            texel_uvs.append(uvs)

        # encode prompt
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt] * batch_size)

        # set timesteps
        self.scheduler.set_timesteps(conf.num_inference_steps)

        # initialize latents from standard normal
        latents = self.prepare_latents(batch_size, generator=generator)

        # denoising loop
        for t in tqdm(self.scheduler.timesteps):
            latents = self.scheduler.scale_model_input(latents, t)

            # denoise first view
            view_0_out = self.model_forward_guided(
                latents[[0]],
                t,
                cond_embeddings[[0]],
                uncond_embeddings[[0]],
                [depth_maps[0]],
                extract_kvs=True,
                extract_qrys=True,
            )

            noise_pred_first = view_0_out.noise_preds[0]

            # extract features
            first_kvs_cond = view_0_out.extracted_kvs_cond
            first_kvs_uncond = view_0_out.extracted_kvs_uncond
            extracted_qrys_cond = view_0_out.extracted_qrys_cond
            extracted_qrys_uncond = view_0_out.extracted_qrys_uncond

            layer_resolutions = map_dict(extracted_qrys_cond, lambda _, x: x.shape[2])
            layer_dimensions = map_dict(extracted_qrys_cond, lambda _, x: x.shape[1])

            # project initial queries to textures
            def initial_texture(layer, features):
                dimension = layer_dimensions[layer]
                uv_map = torch.zeros(uv_res, uv_res, dimension).to(features)
                feature_map = features[0]
                update_uv_texture(uv_map, feature_map, texel_xys[0], texel_uvs[0])
                return uv_map

            textures_cond = map_dict(extracted_qrys_cond, initial_texture)
            textures_uncond = map_dict(extracted_qrys_uncond, initial_texture)

            # denoise remaining views, coniditioned on previous views
            noise_preds = []
            for i in range(1, len(cameras)):
                view_cam = cameras[i]
                view_mesh = meshes[i]

                # render feature textures
                def render_texture(layer, uv_map):
                    tex = make_repeated_uv_texture(uv_map, faces_uvs, verts_uvs)
                    tex.sampling_mode = "nearest"

                    rasterizer = make_mesh_rasterizer(
                        cameras=view_cam, resolution=layer_resolutions[layer]
                    )
                    shader = TextureShader()
                    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

                    render_mesh = view_mesh.clone()
                    render_mesh.textures = tex
                    return renderer(
                        render_mesh,
                    ).to(self.dtype)

                rendered_qrys_cond = map_dict(textures_cond, render_texture)
                rendered_qrys_uncond = map_dict(textures_uncond, render_texture)

                model_out = self.model_forward_guided(
                    latents[[i]],
                    t,
                    cond_embeddings[[i]],
                    uncond_embeddings[[i]],
                    [depth_maps[i]],
                    injected_kvs_cond=first_kvs_cond,  # inject first-view kvs
                    injected_kvs_uncond=first_kvs_uncond,
                    injected_qrys_cond=rendered_qrys_cond,  # inject rendered qrys
                    injected_qrys_uncond=rendered_qrys_uncond,
                    extract_qrys=True,
                )

                noise_pred = model_out.noise_preds[0]
                extracted_qrys_cond = model_out.extracted_qrys_cond
                extracted_qrys_uncond = model_out.extracted_qrys_uncond

                for l in extracted_qrys_cond:
                    update_uv_texture(
                        textures_cond[l],
                        extracted_qrys_cond[l][0],
                        texel_xys[i],
                        texel_uvs[i],
                    )

                    update_uv_texture(
                        textures_uncond[l],
                        extracted_qrys_uncond[l][0],
                        texel_xys[i],
                        texel_uvs[i],
                    )

                noise_preds.append(noise_pred)

            # update latents
            all_noise_preds = [noise_pred_first] + noise_preds
            all_noise_preds = torch.stack(all_noise_preds)
            latents = self.scheduler.step(all_noise_preds, t, latents).prev_sample

        # decode latents
        return self.decode_latents(latents)
