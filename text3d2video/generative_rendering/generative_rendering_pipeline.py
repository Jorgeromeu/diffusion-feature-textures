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
from pytorch3d.renderer import FoVPerspectiveCameras, TexturesVertex
from pytorch3d.structures import Meshes
from torch import Tensor
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from text3d2video.artifacts.gr_data import GrDataArtifact
from text3d2video.generative_rendering.configs import (
    GenerativeRenderingConfig,
    GrSaveConfig,
    NoiseInitializationConfig,
)
from text3d2video.generative_rendering.generative_rendering_attn import (
    GenerativeRenderingAttn,
    GrAttnMode,
)
from text3d2video.rendering import make_feature_renderer, render_depth_map
from text3d2video.sd_feature_extraction import AttnLayerId
from text3d2video.util import (
    aggregate_features_precomputed_vertex_positions,
    project_vertices_to_cameras,
)
from text3d2video.uv_noise import prepare_latents


class GenerativeRenderingPipeline(DiffusionPipeline):
    attn_processor: GenerativeRenderingAttn
    gr_config: GenerativeRenderingConfig
    noise_init_config: NoiseInitializationConfig

    # save config
    gr_save_config = GrSaveConfig
    gr_data_artifact: GrDataArtifact = None

    # indices to save tensors for
    save_frame_indices: List[int]
    save_timesteps: List[int]

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
        frames: Meshes,
        cameras: FoVPerspectiveCameras,
        verts_uvs,
        faces_uvs,
        generator=None,
    ):
        latent_channels = self.unet.config.in_channels
        latent_res = self.gr_config.resolution // 8

        return prepare_latents(
            frames,
            cameras,
            verts_uvs,
            faces_uvs,
            self.noise_init_config,
            latent_channels=latent_channels,
            latent_resolution=latent_res,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

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
        depth_maps: List[Image.Image],
        t: int,
    ) -> Float[Tensor, "b f c h w"]:
        """
        Forward pass of the controlnet and unet
        """

        # batch across time dimension
        chunk_size = latents.shape[0]
        batched_latents = rearrange(latents, "b f c h w -> (b f) c h w")
        batched_embeddings = rearrange(text_embeddings, "b f t d -> (b f) t d")

        # controlnet step
        controlnet_model_input = batched_latents
        controlnet_prompt_embeds = batched_embeddings
        processed_control_image = self.prepare_controlnet_images(depth_maps)
        down_block_res_samples, mid_block_res_sample = self.controlnet(
            controlnet_model_input,
            t,
            encoder_hidden_states=controlnet_prompt_embeds,
            controlnet_cond=processed_control_image,
            conditioning_scale=self.gr_config.controlnet_conditioning_scale,
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

        # unbatch
        noise_pred = rearrange(noise_pred, "(b f) c h w -> b f c h w", b=chunk_size)

        return noise_pred

    def model_forward_feature_extraction(
        self,
        latents: Float[Tensor, "b f c h w"],
        text_embeddings: Float[Tensor, "b f t d"],
        depth_maps: List[Image.Image],
        t: int,
    ) -> Tuple[
        Dict[str, Float[torch.Tensor, "b t c"]],
        Dict[str, Float[torch.Tensor, "b f t c"]],
    ]:
        # set attn processor flags
        self.attn_processor.mode = GrAttnMode.FEATURE_EXTRACTION

        self.attn_processor.pre_attn_features = {}
        self.attn_processor.post_attn_features = {}

        # forward pass
        self.model_forward(latents, text_embeddings, depth_maps, t)

        # get saved features
        pre_attn_features, post_attn_features = (
            self.attn_processor.pre_attn_features,
            self.attn_processor.post_attn_features,
        )

        self.attn_processor.pre_attn_features = {}
        self.attn_processor.post_attn_features = {}

        return pre_attn_features, post_attn_features

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
        self.attn_processor.mode = GrAttnMode.FEATURE_INJECTION
        self.attn_processor.post_attn_features = feature_images
        self.attn_processor.pre_attn_features = pre_attn_features

        noise_pred = self.model_forward(latents, text_embeddings, depth_maps, t)

        return noise_pred

    def sample_keyframe_indices(self, n_frames: int) -> torch.Tensor:
        if self.gr_config.num_keyframes > n_frames:
            raise ValueError("Number of keyframes is greater than number of frames")

        return torch.randperm(n_frames)[: self.gr_config.num_keyframes]

    def aggregate_feature_maps(
        self,
        n_vertices: int,
        kf_vert_xys: List[Tensor],
        kf_vert_indices: List[Tensor],
        saved_post_attn: Dict[str, Float[Tensor, "b f d h w"]],
    ) -> Dict[str, Float[Tensor, "b v d"]]:
        """
        Aggregate features in saved_post_attn across keyframe poses and render them for all poses
        """

        # if not doing post attn injection, skip
        if not self.gr_config.do_post_attn_injection:
            return {}

        all_aggregated_features = {}

        for module, kf_post_attn_features in saved_post_attn.items():
            stacked_vert_features = []
            for feature_maps in kf_post_attn_features:
                # aggregate multi-pose features to 3D
                vert_ft_mean = aggregate_features_precomputed_vertex_positions(
                    feature_maps,
                    n_vertices,
                    kf_vert_xys,
                    kf_vert_indices,
                    mode="bilinear",
                    aggregation_type="mean",
                )

                vert_ft_inpainted = aggregate_features_precomputed_vertex_positions(
                    feature_maps,
                    n_vertices,
                    kf_vert_xys,
                    kf_vert_indices,
                    mode="bilinear",
                    aggregation_type="first",
                )

                w_mean = self.gr_config.mean_features_weight
                w_inpainted = 1 - self.gr_config.mean_features_weight
                vert_features = w_mean * vert_ft_mean + w_inpainted * vert_ft_inpainted

                stacked_vert_features.append(vert_features)

            stacked_vert_features = torch.stack(stacked_vert_features)
            all_aggregated_features[module] = stacked_vert_features

        return all_aggregated_features

    def render_feature_images(
        self,
        cameras: FoVPerspectiveCameras,
        frames: Meshes,
        aggregated_features: Dict[str, Tuple[Float[Tensor, "b v d"], int]],
    ) -> Dict[str, Float[Tensor, "b f d h w"]]:
        """
        render feature images for all modules
        """

        # if not doing post attn injection, skip
        if not self.gr_config.do_post_attn_injection:
            return {}

        all_feature_images = {}

        for module, batched_vert_features in aggregated_features.items():
            # get feature resolution
            attn_layer = AttnLayerId.parse_module_path(module)
            feature_res = attn_layer.layer_resolution(self.unet)

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

    def log_data_artifact(self):
        self.gr_data_artifact.end_recording()
        self.gr_data_artifact.log_if_enabled(delete_folder=False)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        frames: Meshes,
        cameras: FoVPerspectiveCameras,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        generative_rendering_config: GenerativeRenderingConfig,
        noise_initialization_config: NoiseInitializationConfig,
        gr_save_config: GrSaveConfig,
    ):
        # setup configs for use throughout pipeline
        self.gr_config = generative_rendering_config
        self.noise_init_config = noise_initialization_config
        self.gr_save_config = gr_save_config

        # set up attention processor
        self.attn_processor = GenerativeRenderingAttn(
            self.unet, self.gr_config, unet_chunk_size=2
        )
        self.unet.set_attn_processor(self.attn_processor)

        # configure scheduler
        self.scheduler.set_timesteps(self.gr_config.num_inference_steps)
        n_frames = len(frames)

        # setup save tensors
        gr_artifact = GrDataArtifact.init_from_config(gr_save_config)
        self.gr_data_artifact = gr_artifact
        self.attn_processor.gr_data_artifact = gr_artifact
        self.gr_data_artifact.begin_recording(self.scheduler, n_frames)

        # setup generator
        generator = torch.Generator(device=self.device)
        generator.manual_seed(self.gr_config.seed)

        # render depth maps for frames
        depth_maps = render_depth_map(frames, cameras, self.gr_config.resolution)

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt] * n_frames)
        stacked_text_embeddings = torch.stack([uncond_embeddings, cond_embeddings])

        # initial latent noise
        latents = self.prepare_latents(frames, cameras, verts_uvs, faces_uvs, generator)

        # chunk indices to use in inference loop
        chunks_indices = torch.split(
            torch.arange(0, n_frames), self.gr_config.chunk_size
        )

        # get 2D vertex positions for each frame
        vert_xys, vert_indices = project_vertices_to_cameras(frames, cameras)

        # denoising loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):
            self.gr_data_artifact.save_latents(t, latents)

            # update timestep
            self.attn_processor.cur_timestep = t

            # duplicate latent, for classifier-free guidance
            latents_stacked = torch.stack([latents] * 2)
            latents_stacked = self.scheduler.scale_model_input(latents_stacked, t)

            # sample keyframe indices
            kf_indices = self.sample_keyframe_indices(n_frames)

            # Diffusion step #1 on keyframes, to extract features
            kf_latents = latents_stacked[:, kf_indices]
            kf_embeddings = stacked_text_embeddings[:, kf_indices]
            kf_depth_maps = [depth_maps[i] for i in kf_indices.tolist()]

            pre_attn_features, post_attn_features = (
                self.model_forward_feature_extraction(
                    kf_latents,
                    kf_embeddings,
                    kf_depth_maps,
                    t,
                )
            )

            # save kf post attn features and indices
            self.gr_data_artifact.gr_writer.write_kf_indices(t, kf_indices)
            self.gr_data_artifact.gr_writer.write_kf_post_attn(t, post_attn_features)

            # unify spatial features across keyframes as vertex features
            kf_vert_xys = [vert_xys[i] for i in kf_indices.tolist()]
            kf_vert_indices = [vert_indices[i] for i in kf_indices.tolist()]
            aggregated_3d_features = self.aggregate_feature_maps(
                frames.num_verts_per_mesh()[0],
                kf_vert_xys,
                kf_vert_indices,
                post_attn_features,
            )

            # save aggregated features
            self.gr_data_artifact.gr_writer.write_vertex_features(
                t, aggregated_3d_features
            )

            # do inference in chunks
            noise_preds = []
            for chunk_indices in tqdm(chunks_indices, desc="Chunks"):
                # render chunk feature images
                chunk_feature_images = self.render_feature_images(
                    cameras[chunk_indices],
                    frames[chunk_indices],
                    aggregated_3d_features,
                )

                # Diffusion step #2 with pre and post attn feature injection
                # get chunk inputs
                chunk_latents = latents_stacked[:, chunk_indices]
                chunk_embeddings = stacked_text_embeddings[:, chunk_indices]
                chunk_depth_maps = [depth_maps[i] for i in chunk_indices.tolist()]

                self.attn_processor.chunk_indices = chunk_indices
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
            noise_pred = (
                noise_pred_uncond + self.gr_config.guidance_scale * guidance_direction
            )

            # update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

        self.gr_data_artifact.save_latents(0, latents)

        # decode latents in chunks
        decoded_imgs = []
        for chunk_indices in chunks_indices:
            chunk_latents = latents[chunk_indices]
            chunk_images = self.latents_to_images(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        self.gr_data_artifact.end_recording()
        return decoded_imgs
