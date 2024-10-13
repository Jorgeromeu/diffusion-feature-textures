from math import sqrt
from typing import Dict, List, Tuple

import rerun as rr
import torch
from diffusers import (
    AutoencoderKL,
    ControlNetModel,
    DiffusionPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
)
from diffusers.image_processor import VaeImageProcessor
from einops import rearrange
from jaxtyping import Float
from PIL import Image
from pytorch3d.renderer import FoVPerspectiveCameras, TexturesUV, TexturesVertex
from pytorch3d.structures import Meshes
from torch import Tensor
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from typeguard import typechecked

from scripts.aggregate_features import aggregate_3d_features
from text3d2video.generative_rendering.generative_rendering_attn import (
    GenerativeRenderingAttn,
)
from text3d2video.rendering import make_feature_renderer, render_depth_map


class GenerativeRenderingPipeline(DiffusionPipeline):

    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    num_keyframes: int = 2
    do_uv_noise_init: bool = True
    do_pre_attn_injection: bool = True
    do_post_attn_injection: bool = True
    chunk_size: int = 10
    module_paths: list[str]

    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet2DConditionModel,
        scheduler: UniPCMultistepScheduler,
        controlnet: ControlNetModel,
    ):

        super().__init__()

        # register modules
        self.register_modules(
            vae=vae,
            unet=unet,
            scheduler=scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            controlnet=controlnet,
        )

        # vae image processors
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True
        )
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor,
            do_convert_rgb=True,
            do_normalize=False,
        )

    def encode_prompt(self, prompts: List[str]) -> Tuple[Tensor, Tensor]:

        # tokenize prompts
        text_input = self.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )

        # Get CLIP embedding
        with torch.no_grad():
            cond_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
            [""] * len(prompts),
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        return cond_embeddings, uncond_embeddings

    def prepare_latents_random(
        self, batch_size: int, out_resolution: int, generator=None
    ) -> Float[Tensor, "b c h w"]:
        latent_res = out_resolution // 8
        in_channels = self.unet.config.in_channels
        latents = torch.randn(
            batch_size,
            in_channels,
            latent_res,
            latent_res,
            device=self.device,
            generator=generator,
            dtype=self.dtype,
        )

        return latents

    def prepare_uv_initialized_latents(
        self,
        frames: Meshes,
        cameras: FoVPerspectiveCameras,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        out_resolution: int = 512,
        generator=None,
    ) -> Float[Tensor, "b c h w"]:

        # setup noise texture
        latent_res = out_resolution // 8
        noise_texture_res = latent_res * 1
        in_channels = self.unet.config.in_channels
        noise_texture_map = torch.randn(
            noise_texture_res,
            noise_texture_res,
            in_channels,
            device=self.device,
            generator=generator,
        )

        n_frames = len(frames)

        noise_texture = TexturesUV(
            verts_uvs=verts_uvs.expand(n_frames, -1, -1).to(self.device),
            faces_uvs=faces_uvs.expand(n_frames, -1, -1).to(self.device),
            maps=noise_texture_map.expand(n_frames, -1, -1, -1).to(self.device),
        )

        frames.textures = noise_texture

        # render noise texture for each frame
        renderer = make_feature_renderer(cameras, latent_res)
        noise_renders = renderer(frames).to(self.dtype)

        noise_renders = rearrange(noise_renders, "b h w c -> b c h w")
        noise_renders = noise_renders.to(device=self.device, dtype=self.dtype)

        background_noise = torch.randn(
            in_channels,
            latent_res,
            latent_res,
        ).expand(n_frames, -1, -1, -1)
        background_noise = background_noise.to(self.device, dtype=self.dtype)

        latents_mask = (noise_renders == 0).float()
        latents = noise_renders + background_noise * latents_mask

        latents = latents.to(self.device, dtype=self.dtype)

        return latents

    def latents_to_images(self, latents: torch.FloatTensor, generator=None):

        # scale latents
        latents_scaled = latents / self.vae.config.scaling_factor

        # decode latents
        images = self.vae.decode(
            latents_scaled,
            return_dict=False,
            generator=generator,
        )[0]

        # postprocess images
        images = self.image_processor.postprocess(
            images, output_type="pil", do_denormalize=[True] * len(latents)
        )

        return images

    def prepare_controlnet_image(
        self, images: List[Image.Image], do_classifier_free_guidance=True
    ):

        height = images[0].height
        width = images[0].width

        image = self.control_image_processor.preprocess(
            images, height=height, width=width
        ).to(dtype=self.dtype, device=self.device)

        if do_classifier_free_guidance:
            image = torch.cat([image] * 2)

        return image

    def controlnet_and_unet_forward(
        self,
        latents: torch.Tensor,
        text_embeddings: torch.Tensor,
        depth_maps: List[Image.Image],
        t: int,
    ):

        # controlnet step
        controlnet_model_input = latents
        controlnet_prompt_embeds = text_embeddings
        processed_control_image = self.prepare_controlnet_image(depth_maps)
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            controlnet_model_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=processed_control_image,
            conditioning_scale=self.controlnet_conditioning_scale,
            guess_mode=False,
            return_dict=False,
        )

        # unet, with controlnet residuals
        return self.unet(
            latents,
            t,
            mid_block_additional_residual=mid_block_res_sample,
            down_block_additional_residuals=down_block_res_samples,
            encoder_hidden_states=text_embeddings,
        )

    def model_forward(
        self,
        latents: Float[Tensor, "b f c h w"],
        text_embeddings: Float[Tensor, "b f t d"],
        depth_maps: List[Image.Image],
        t: int,
    ) -> Float[Tensor, "b f c h w"]:

        chunk_size = latents.shape[0]
        batched_latents = rearrange(latents, "b f c h w -> (b f) c h w")
        batched_embeddings = rearrange(text_embeddings, "b f t d -> (b f) t d")

        # controlnet step
        controlnet_model_input = batched_latents
        controlnet_prompt_embeds = batched_embeddings
        processed_control_image = self.prepare_controlnet_image(depth_maps)
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            controlnet_model_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=processed_control_image,
            guess_mode=False,
            return_dict=False,
        )

        # unet, with controlnet residuals
        noise_pred = self.unet(
            batched_latents,
            t,
            mid_block_additional_residual=mid_block_res_sample,
            down_block_additional_residuals=down_block_res_samples,
            encoder_hidden_states=batched_embeddings,
        ).sample

        noise_pred = rearrange(noise_pred, "(b f) c h w -> b f c h w", b=chunk_size)
        return noise_pred

    def model_forward_feature_extraction(
        self,
        latents: Float[Tensor, "b f c h w"],
        text_embeddings: Float[Tensor, "b f t d"],
        depth_maps: List[Image.Image],
        t: int,
    ) -> Float[Tensor, "b f c h w"]:

        # do extended attention and save features
        self.attn_processor.do_extended_attention = True
        self.attn_processor.save_pre_attn_features = self.do_pre_attn_injection
        self.attn_processor.save_post_attn_features = self.do_post_attn_injection

        self.model_forward(latents, text_embeddings, depth_maps, t)

        self.attn_processor.save_pre_attn_features = False
        self.attn_processor.save_post_attn_features = False
        self.attn_processor.do_extended_attention = False

        return self.attn_processor.saved_pre_attn, self.attn_processor.saved_post_attn

    def model_forward_feature_injection(
        self,
        latents: Float[Tensor, "b f c h w"],
        text_embeddings: Float[Tensor, "b f t d"],
        depth_maps: List[Image.Image],
        t: int,
        pre_attn_features: Dict[str, Float[Tensor, "b f t d"]],
        feature_images: Dict[str, Float[Tensor, "b f d h w"]],
    ):

        # pass features to attn processor
        self.attn_processor.feature_images = feature_images
        self.attn_processor.saved_pre_attn = pre_attn_features

        self.attn_processor.save_pre_attn_features = self.do_pre_attn_injection
        self.attn_processor.save_post_attn_features = self.do_post_attn_injection
        self.attn_processor.feature_blend_alpha = self.feature_blend_alpha

        noise_pred = self.model_forward(latents, text_embeddings, depth_maps, t)

        self.attn_processor.do_pre_attn_injection = False
        self.attn_processor.do_post_attn_injection = False

        return noise_pred

    def sample_keyframes(self, n_frames: int):
        return torch.randperm(n_frames)[: self.num_keyframes]

    def aggregate_kf_features(
        self,
        kf_cameras: FoVPerspectiveCameras,
        kf_frames: Meshes,
        saved_post_attn: Dict[str, Float[Tensor, "b f t d"]],
    ) -> Dict[str, Tuple[Float[Tensor, "b v d"], int]]:
        """
        Aggregate features in saved_post_attn across keyframe poses and render them for all poses
        """

        # if not doing post attn injection, skip
        if not self.do_post_attn_injection:
            return {}

        all_aggregated_features = {}

        for module, kf_post_attn_features in saved_post_attn.items():

            # reshape features to square
            feature_res = int(sqrt(kf_post_attn_features.shape[2]))
            kf_post_attn_feature_maps = rearrange(
                kf_post_attn_features,
                "b f_kf (h w) d -> b f_kf d h w",
                h=feature_res,
                f_kf=len(kf_cameras),
            )

            stacked_vert_features = []
            for feature_maps in kf_post_attn_feature_maps:
                # aggregate multi-pose features to 3D
                vert_features = aggregate_3d_features(
                    kf_cameras, kf_frames, feature_maps
                )
                stacked_vert_features.append(vert_features)

            stacked_vert_features = torch.stack(stacked_vert_features)
            all_aggregated_features[module] = (stacked_vert_features, feature_res)

        return all_aggregated_features

    def render_aggregated_features(
        self,
        cameras: FoVPerspectiveCameras,
        frames: Meshes,
        aggregated_features: Dict[str, Tuple[Float[Tensor, "b v d"], int]],
    ) -> Dict[str, Float[Tensor, "b f d h w"]]:
        """
        Aggregate features in saved_post_attn across keyframe poses and render them for all poses
        """

        # if not doing post attn injection, skip
        if not self.do_post_attn_injection:
            return {}

        all_feature_images = {}

        for module, (batched_vert_features, feature_res) in aggregated_features.items():

            renderer = make_feature_renderer(cameras, feature_res)

            # render for each batch
            stacked_feature_images = []
            for vert_features in batched_vert_features:

                # construct feature texture
                vert_features_tex = TexturesVertex(
                    vert_features.expand(len(frames), -1, -1).to(self.device)
                )
                frames.textures = vert_features_tex
                feature_images = renderer(frames)
                feature_images = rearrange(feature_images, "f h w d -> f d h w")
                stacked_feature_images.append(feature_images)

            # B, F, D, H, W
            stacked_feature_images = torch.stack(stacked_feature_images)
            all_feature_images[module] = stacked_feature_images

        return all_feature_images

    def aggregate_and_render_kf_features(
        self,
        cameras: FoVPerspectiveCameras,
        frames: Meshes,
        kf_cameras: FoVPerspectiveCameras,
        kf_frames: Meshes,
        saved_post_attn: Dict[str, Float[Tensor, "b f t d"]],
    ) -> Dict[str, Float[Tensor, "b c h w d"]]:
        """
        Aggregate features in saved_post_attn across keyframe poses and render them for all poses
        """

        # if not doing post attn injection, skip
        if not self.do_post_attn_injection:
            return {}

        all_feature_images = {}

        for module, kf_post_attn_features in saved_post_attn.items():

            # reshape features to square
            feature_res = int(sqrt(kf_post_attn_features.shape[2]))
            kf_post_attn_feature_maps = rearrange(
                kf_post_attn_features,
                "b f_kf (h w) d -> b f_kf d h w",
                h=feature_res,
                f_kf=len(kf_cameras),
            )

            renderer = make_feature_renderer(cameras, feature_res)

            # aggregate and render for each batch
            stacked_feature_images = []
            for feature_maps in kf_post_attn_feature_maps:

                # aggregate multi-pose features to 3D
                vert_features = aggregate_3d_features(
                    kf_cameras, kf_frames, feature_maps
                )

                # construct feature texture
                vert_features_tex = TexturesVertex(
                    vert_features.expand(len(frames), -1, -1).to(self.device)
                )
                frames.textures = vert_features_tex
                feature_images = renderer(frames)
                feature_images = rearrange(feature_images, "f h w d -> f d h w")
                stacked_feature_images.append(feature_images)

            # B, F, D, H, W
            stacked_feature_images = torch.stack(stacked_feature_images)
            all_feature_images[module] = stacked_feature_images

        return all_feature_images

    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    @torch.no_grad()
    @typechecked
    def __call__(
        self,
        prompt: str,
        frames: Meshes,
        cameras: FoVPerspectiveCameras,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        res: int,
        num_inference_steps: int,
        do_uv_noise_init: bool,
        do_pre_attn_injection: bool,
        do_post_attn_injection: bool,
        feature_blend_alpha: float,
        num_keyframes: int,
        guidance_scale: float,
        controlnet_conditioning_scale: float,
        chunk_size: int,
        module_paths: List[str],
        seed: int,
    ):

        self.update_params(
            feature_blend_alpha=feature_blend_alpha,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            num_keyframes=num_keyframes,
            do_uv_noise_init=do_uv_noise_init,
            do_pre_attn_injection=do_pre_attn_injection,
            do_post_attn_injection=do_post_attn_injection,
            chunk_size=chunk_size,
            module_paths=module_paths,
            seed=seed,
        )

        # setup generator
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        n_frames = len(frames)

        # render depth maps
        depth_maps = render_depth_map(frames, cameras, res)

        for i, depth_map in enumerate(depth_maps):
            rr.log(f"depth_map_{i}", rr.Image(depth_map))

        # Get prompt embeddings for guidance
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt] * n_frames)
        stacked_text_embeddings = torch.stack([uncond_embeddings, cond_embeddings])

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # initial latent noise
        if do_uv_noise_init:
            latents = self.prepare_uv_initialized_latents(
                frames, cameras, verts_uvs, faces_uvs
            )
        else:
            latents = self.prepare_latents_random(n_frames, res, generator=generator)

        # setup attn processor
        self.attn_processor = GenerativeRenderingAttn(self.unet, unet_chunk_size=2)
        self.attn_processor.module_paths = self.module_paths
        self.unet.set_attn_processor(self.attn_processor)

        # chunk indices to use
        chunks_indices = torch.split(torch.arange(0, n_frames), chunk_size)

        # denoising loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):

            # duplicate latent, to feed to model with CFG
            # 2, F, C, H, W
            latents_stacked = torch.stack([latents] * 2)
            latents_stacked = self.scheduler.scale_model_input(latents_stacked, t)

            # sample keyframe indices
            kf_indices = self.sample_keyframes(n_frames)

            # get keyframe inputs
            kf_latents = latents_stacked[:, kf_indices]
            kf_embeddings = stacked_text_embeddings[:, kf_indices]
            kf_depth_maps = [depth_maps[i] for i in kf_indices.tolist()]

            # Diffusion step #1 on keyframes, to extract features
            pre_attn_features, post_attn_features = (
                self.model_forward_feature_extraction(
                    kf_latents,
                    kf_embeddings,
                    kf_depth_maps,
                    t,
                )
            )

            # aggregate keyframe features
            aggregated_3d_features = self.aggregate_kf_features(
                cameras[kf_indices], frames[kf_indices], post_attn_features
            )

            # do inference in chunks
            noise_preds = []

            self.attn_processor.do_pre_attn_injection = do_pre_attn_injection
            self.attn_processor.do_post_attn_injection = do_post_attn_injection

            for chunk_indices in tqdm(chunks_indices, desc="Chunks"):

                # get chunk inputs
                chunk_latents = latents_stacked[:, chunk_indices]
                chunk_embeddings = stacked_text_embeddings[:, chunk_indices]
                chunk_depth_maps = [depth_maps[i] for i in chunk_indices.tolist()]

                # render chunk feature images
                chunk_feature_images = self.render_aggregated_features(
                    cameras[chunk_indices],
                    frames[chunk_indices],
                    aggregated_3d_features,
                )

                # Diffusion step 2 with feature injection
                noise_pred = self.model_forward_feature_injection(
                    chunk_latents,
                    chunk_embeddings,
                    chunk_depth_maps,
                    t,
                    pre_attn_features,
                    chunk_feature_images,
                )

                noise_preds.append(noise_pred)

            # concatenate predictions
            noise_pred_all = torch.cat(noise_preds, dim=1)

            # preform classifier free guidance
            noise_pred_uncond, noise_pred_text = noise_pred_all
            guidance_direction = noise_pred_text - noise_pred_uncond
            noise_pred = noise_pred_uncond + guidance_scale * guidance_direction

            # update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        # decode latents in chunks
        decoded_imgs = []
        for chunk_indices in chunks_indices:
            chunk_latents = latents[chunk_indices]
            chunk_images = self.latents_to_images(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        return decoded_imgs
