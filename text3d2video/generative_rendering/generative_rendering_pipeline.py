from typing import Dict, List, Tuple

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
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
)
from pytorch3d.structures import Meshes
from torch import Tensor
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from text3d2video.artifacts.gr_data import GrDataArtifact, GrSaveConfig
from text3d2video.backprojection import (
    aggregate_spatial_features_dict,
    project_visible_verts_to_cameras,
    rasterize_and_render_vert_features_dict,
)
from text3d2video.generative_rendering.configs import (
    GenerativeRenderingConfig,
)
from text3d2video.generative_rendering.generative_rendering_attn import (
    GenerativeRenderingAttn,
    GrAttnMode,
)
from text3d2video.noise_initialization import NoiseInitializer
from text3d2video.rendering import render_depth_map


class GenerativeRenderingPipeline(DiffusionPipeline):
    attn_processor: GenerativeRenderingAttn
    rd_config: GenerativeRenderingConfig
    noise_initializer: NoiseInitializer

    # diffusion data artifact
    gr_data_artifact: GrDataArtifact = None

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

    def prepare_latents(
        self,
        meshes: Meshes,
        cameras: FoVPerspectiveCameras,
        verts_uvs,
        faces_uvs,
        generator=None,
    ):
        return self.noise_initializer.initial_noise(
            cameras=cameras,
            meshes=meshes,
            verts_uvs=verts_uvs,
            faces_uvs=faces_uvs,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
            n_frames=len(meshes),
        )

    def decode_latents(self, latents: torch.FloatTensor, generator=None):
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

    def prepare_controlnet_images(
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

    def model_forward(
        self,
        latents: Float[Tensor, "b f c h w"],
        text_embeddings: Float[Tensor, "b f t d"],
        t: int,
        depth_maps: List[Image.Image],
    ) -> Float[Tensor, "b f c h w"]:
        """
        Forward pass of the controlnet and unet
        """

        # batch across time dimension
        chunk_size = latents.shape[0]
        batched_latents = rearrange(latents, "b f c h w -> (b f) c h w")
        batched_embeddings = rearrange(text_embeddings, "b f t d -> (b f) t d")

        if depth_maps:
            # controlnet step
            controlnet_model_input = batched_latents
            controlnet_prompt_embeds = batched_embeddings
            processed_control_image = self.prepare_controlnet_images(depth_maps)
            down_block_res_samples, mid_block_res_sample = self.controlnet(
                controlnet_model_input,
                t,
                encoder_hidden_states=controlnet_prompt_embeds,
                controlnet_cond=processed_control_image,
                conditioning_scale=self.rd_config.controlnet_conditioning_scale,
                guess_mode=False,
                return_dict=False,
            )
        else:
            down_block_res_samples, mid_block_res_sample = None, None

        # unet, with controlnet residuals
        noise_pred = self.unet(
            batched_latents,
            t,
            mid_block_additional_residual=mid_block_res_sample,
            down_block_additional_residuals=down_block_res_samples,
            encoder_hidden_states=batched_embeddings,
        ).sample

        # unbatch
        noise_pred = rearrange(noise_pred, "(b f) c h w -> b f c h w", b=chunk_size)

        return noise_pred

    def model_fwd_feature_extraction(
        self,
        latents: Float[Tensor, "b f c h w"],
        text_embeddings: Float[Tensor, "b f t d"],
        depth_maps: List[Image.Image],
        t: int,
    ) -> Tuple[
        Dict[str, Float[torch.Tensor, "b t c"]],
        Dict[str, Float[torch.Tensor, "b f t c"]],
    ]:
        # set attn processor mode
        self.attn_processor.mode = GrAttnMode.FEATURE_EXTRACTION

        # clear saved features
        self.attn_processor.pre_attn_features = {}
        self.attn_processor.post_attn_features = {}

        # forward pass
        self.model_forward(latents, text_embeddings, t, depth_maps=depth_maps)

        # get saved features
        pre_attn_features, post_attn_features = (
            self.attn_processor.pre_attn_features,
            self.attn_processor.post_attn_features,
        )

        # clear saved features
        self.attn_processor.pre_attn_features = {}
        self.attn_processor.post_attn_features = {}

        return pre_attn_features, post_attn_features

    def model_fwd_feature_injection(
        self,
        latents: Float[Tensor, "b f c h w"],
        text_embeddings: Float[Tensor, "b f t d"],
        depth_maps: List[Image.Image],
        t: int,
        pre_attn_features: Dict[str, Float[Tensor, "b f t d"]],
        feature_images: Dict[str, Float[Tensor, "b f d h w"]],
        frame_indices: Tensor,
    ):
        # set attn processor mode
        self.attn_processor.mode = GrAttnMode.FEATURE_INJECTION

        # pass features to attn processor
        self.attn_processor.post_attn_features = feature_images
        self.attn_processor.pre_attn_features = pre_attn_features

        # pass frame indices to attn processor
        self.attn_processor.set_chunk_frame_indices(frame_indices)
        noise_pred = self.model_forward(
            latents, text_embeddings, t, depth_maps=depth_maps
        )

        return noise_pred

    def sample_keyframe_indices(self, n_frames: int) -> torch.Tensor:
        if self.rd_config.num_keyframes > n_frames:
            raise ValueError("Number of keyframes is greater than number of frames")
        return torch.randperm(n_frames)[: self.rd_config.num_keyframes]

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        meshes: Meshes,
        cameras: FoVPerspectiveCameras,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        generative_rendering_config: GenerativeRenderingConfig,
        noise_initializer: NoiseInitializer,
        gr_save_config: GrSaveConfig,
    ):
        # setup configs for use throughout pipeline
        self.rd_config = generative_rendering_config
        self.noise_initializer = noise_initializer

        # set up attention processor
        self.attn_processor = GenerativeRenderingAttn(
            self.unet, self.rd_config, unet_chunk_size=2
        )
        self.unet.set_attn_processor(self.attn_processor)

        # configure scheduler
        self.scheduler.set_timesteps(self.rd_config.num_inference_steps)
        n_frames = len(meshes)

        # setup diffusion data
        data_artifact = GrDataArtifact.init_from_config(gr_save_config)
        self.gr_data_artifact = data_artifact
        self.attn_processor.gr_data_artifact = data_artifact
        self.gr_data_artifact.begin_recording(self.scheduler, n_frames)

        # setup generator
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.rd_config.seed)

        # render depth maps for frames
        depth_maps = render_depth_map(meshes, cameras, self.rd_config.resolution)

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt] * n_frames)
        stacked_text_embeddings = torch.stack([uncond_embeddings, cond_embeddings])

        # initial latent noise
        latents = self.prepare_latents(meshes, cameras, verts_uvs, faces_uvs, generator)

        # chunk indices to use in inference loop
        chunks_indices = torch.split(
            torch.arange(0, n_frames), self.rd_config.chunk_size
        )

        # get 2D vertex positions for each frame
        vert_xys, vert_indices = project_visible_verts_to_cameras(meshes, cameras)

        # denoising loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            self.gr_data_artifact.latents_writer.write_latents_batched(t, latents)

            # update timestep
            self.attn_processor.cur_timestep = t

            # duplicate latent, for classifier-free guidance
            latents_stacked = torch.stack([latents] * 2)
            latents_stacked = self.scheduler.scale_model_input(latents_stacked, t)

            # sample keyframe indices
            kf_indices = self.sample_keyframe_indices(n_frames)

            # Feature Extraction Step
            kf_latents = latents_stacked[:, kf_indices]
            kf_embeddings = stacked_text_embeddings[:, kf_indices]
            kf_depth_maps = [depth_maps[i] for i in kf_indices.tolist()]

            pre_attn_features, post_attn_features = self.model_fwd_feature_extraction(
                kf_latents,
                kf_embeddings,
                kf_depth_maps,
                t,
            )

            # save kf post attn features and indices
            self.gr_data_artifact.gr_writer.write_kf_indices(t, kf_indices)
            self.gr_data_artifact.gr_writer.write_kf_post_attn(t, post_attn_features)

            # unify spatial features across keyframes as vertex features
            kf_vert_xys = [vert_xys[i] for i in kf_indices.tolist()]
            kf_vert_indices = [vert_indices[i] for i in kf_indices.tolist()]

            aggregated_3d_features = aggregate_spatial_features_dict(
                post_attn_features,
                meshes.num_verts_per_mesh()[0],
                kf_vert_xys,
                kf_vert_indices,
            )

            layer_resolutions = {
                layer: feature.shape[-1]
                for layer, feature in post_attn_features.items()
            }

            # save aggregated features
            self.gr_data_artifact.gr_writer.write_vertex_features(
                t, aggregated_3d_features
            )

            # do inference in chunks
            noise_preds = []
            for i, chunk_frame_indices in enumerate(chunks_indices):
                chunk_feature_images = rasterize_and_render_vert_features_dict(
                    aggregated_3d_features,
                    meshes[chunk_frame_indices],
                    cameras[chunk_frame_indices],
                    resolutions=layer_resolutions,
                )

                # Diffusion step #2 with pre and post attn feature injection
                # get chunk inputs
                chunk_latents = latents_stacked[:, chunk_frame_indices]
                chunk_embeddings = stacked_text_embeddings[:, chunk_frame_indices]
                chunk_depth_maps = [depth_maps[i] for i in chunk_frame_indices.tolist()]

                noise_pred = self.model_fwd_feature_injection(
                    chunk_latents,
                    chunk_embeddings,
                    chunk_depth_maps,
                    t,
                    pre_attn_features,
                    chunk_feature_images,
                    chunk_frame_indices,
                )
                noise_preds.append(noise_pred)

            # concatenate predictions
            noise_pred_all = torch.cat(noise_preds, dim=1)

            # preform classifier free guidance
            noise_pred_uncond, noise_pred_text = noise_pred_all
            guidance_direction = noise_pred_text - noise_pred_uncond
            noise_pred = (
                noise_pred_uncond + self.rd_config.guidance_scale * guidance_direction
            )

            # update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        self.gr_data_artifact.latents_writer.write_latents_batched(0, latents)

        # decode latents in chunks
        decoded_imgs = []
        for chunk_frame_indices in chunks_indices:
            chunk_latents = latents[chunk_frame_indices]
            chunk_images = self.decode_latents(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        self.gr_data_artifact.end_recording()
        return decoded_imgs
