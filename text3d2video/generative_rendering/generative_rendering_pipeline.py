from typing import Dict, List, Tuple

import rerun as rr
import rerun.blueprint as rrb
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

import text3d2video.rerun_util as ru
from text3d2video.artifacts.tensors_artifact import TensorsArtifact
from text3d2video.feature_visualization import RgbPcaUtil
from text3d2video.generative_rendering.generative_rendering_attn import (
    GenerativeRenderingAttn,
)
from text3d2video.rendering import make_feature_renderer, render_depth_map
from text3d2video.util import (
    aggregate_features_precomputed_vertex_positions,
    ordered_sample,
    project_vertices_to_cameras,
)


class GenerativeRenderingPipeline(DiffusionPipeline):

    # rerun settings
    rerun: bool
    rerun_module_paths: List[str]
    rerun_frame_indices: List[int]

    # model settings
    res: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    controlnet_conditioning_scale: float = 1.0
    num_keyframes: int = 2
    do_uv_noise_init: bool = True
    do_pre_attn_injection: bool = True
    do_post_attn_injection: bool = True
    chunk_size: int = 10
    module_paths: list[str]
    mean_features_weight: float = 0.5

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

        # setup attn processor
        self.attn_processor = GenerativeRenderingAttn(self.unet, unet_chunk_size=2)
        self.unet.set_attn_processor(self.attn_processor)

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

    def prepare_random_latents(
        self, batch_size: int, generator=None
    ) -> Float[Tensor, "b c w h"]:
        latent_res = self.res // 8
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
        generator=None,
    ) -> Float[Tensor, "b c w h"]:

        # setup noise texture
        latent_res = self.res // 8
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
            in_channels, latent_res, latent_res, generator=generator, device=self.device
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

        # do extended attention and save features
        self.attn_processor.do_extended_attention = True
        self.attn_processor.save_pre_attn_features = self.do_pre_attn_injection
        self.attn_processor.save_post_attn_features = self.do_post_attn_injection
        self.attn_processor.module_paths = self.module_paths

        self.model_forward(latents, text_embeddings, depth_maps, t)

        # clear flags
        self.attn_processor.save_pre_attn_features = False
        self.attn_processor.save_post_attn_features = False
        self.attn_processor.do_extended_attention = False
        self.attn_processor.module_paths = []

        # get saved features
        pre_attn_features, post_attn_features = (
            self.attn_processor.saved_pre_attn,
            self.attn_processor.saved_post_attn,
        )

        # clear saved features
        self.attn_processor.saved_pre_attn = {}
        self.attn_processor.saved_post_attn = {}

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

        # set modules
        self.attn_processor.module_paths = self.module_paths

        # pass features to attn processor
        self.attn_processor.feature_images = feature_images
        self.attn_processor.saved_pre_attn = pre_attn_features

        # do feature injection
        self.attn_processor.do_pre_attn_injection = self.do_pre_attn_injection
        self.attn_processor.do_post_attn_injection = self.do_post_attn_injection
        self.attn_processor.feature_blend_alpha = self.feature_blend_alpha

        noise_pred = self.model_forward(latents, text_embeddings, depth_maps, t)

        self.attn_processor.module_paths = []
        self.attn_processor.feature_images = {}
        self.attn_processor.saved_pre_attn = {}
        self.attn_processor.do_pre_attn_injection = False
        self.attn_processor.do_post_attn_injection = False

        return noise_pred

    def sample_keyframes(self, n_frames: int) -> torch.Tensor:

        if self.num_keyframes > n_frames:
            raise ValueError("Number of keyframes is greater than number of frames")

        return torch.randperm(n_frames)[: self.num_keyframes]

    def aggregate_kf_features(
        self,
        n_vertices: int,
        kf_vert_xys: List[Tensor],
        kf_vert_indices: List[Tensor],
        saved_post_attn: Dict[str, Float[Tensor, "b f d h w"]],
    ) -> Dict[str, Tuple[Float[Tensor, "b v d"], int]]:
        """
        Aggregate features in saved_post_attn across keyframe poses and render them for all poses
        """

        # if not doing post attn injection, skip
        if not self.do_post_attn_injection:
            return {}

        all_aggregated_features = {}

        for module, kf_post_attn_features in saved_post_attn.items():

            feature_map_res = kf_post_attn_features.shape[-1]

            stacked_vert_features = []
            for feature_maps in kf_post_attn_features:

                # aggregate multi-pose features to 3D
                vert_ft_mean = aggregate_features_precomputed_vertex_positions(
                    feature_maps,
                    n_vertices,
                    kf_vert_xys,
                    kf_vert_indices,
                    aggregation_type="mean",
                )

                vert_ft_inpainted = aggregate_features_precomputed_vertex_positions(
                    feature_maps,
                    n_vertices,
                    kf_vert_xys,
                    kf_vert_indices,
                    aggregation_type="first",
                )

                w_mean = self.mean_features_weight
                w_inpainted = 1 - self.mean_features_weight
                vert_features = w_mean * vert_ft_mean + w_inpainted * vert_ft_inpainted

                stacked_vert_features.append(vert_features)

            stacked_vert_features = torch.stack(stacked_vert_features)
            all_aggregated_features[module] = (stacked_vert_features, feature_map_res)

        return all_aggregated_features

    def render_feature_images(
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

    def update_params(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def setup_rerun_blueprint(self, n_frames: int):

        frame_views = []
        latent_views = []
        depth_map_views = []

        for frame_i in self.rerun_frame_indices:
            frame_views.append(rrb.Spatial2DView(contents=[f"+/frame_{frame_i}"]))
            latent_views.append(rrb.TensorView(contents=[f"+/latent_{frame_i}"]))
            depth_map_views.append(
                rrb.Spatial2DView(contents=[f"+/depth_map_{frame_i}"])
            )

        main_tab = rrb.Vertical(
            rrb.Horizontal(*latent_views, name="Latents"),
            rrb.Horizontal(*frame_views, name="Frames"),
            rrb.Horizontal(*depth_map_views, name="Depth Maps"),
            name="Generated Images",
        )

        attn_out_views = []
        rendered_views = []
        blended_views = []

        for frame_i in self.rerun_frame_indices:
            attn_out_views.append(
                rrb.Spatial2DView(contents=[f"+/attn_out_{frame_i}"], name="Attn Out"),
            )
            rendered_views.append(
                rrb.Spatial2DView(contents=[f"+/rendered_{frame_i}"], name="Rendered")
            )
            blended_views.append(
                rrb.Spatial2DView(contents=[f"+/blended_{frame_i}"], name="Blended")
            )

        rendered_features_tab = rrb.Horizontal(
            rrb.Spatial3DView(name="3D", contents=["+/mesh"]),
            rrb.Vertical(
                rrb.Horizontal(*attn_out_views, name="Attn Out"),
                rrb.Horizontal(*rendered_views, name="Rendered"),
                rrb.Horizontal(*blended_views, name="Blended"),
            ),
        )

        return rrb.Blueprint(
            rrb.Tabs(main_tab, rendered_features_tab),
            collapse_panels=True,
        )

    def setup_rerun(self, n_frames: int):

        # init rerun
        ru.set_logging_state(self.rerun)
        rr.init("Generative Rendering")
        rr.serve()

        # log a maximum of 5 frames
        self.rerun_frame_indices = list(range(n_frames))
        if n_frames > 5:
            self.rerun_frame_indices = ordered_sample(self.rerun_frame_indices, 5)

        # setup blueprint
        if self.rerun:
            rr.send_blueprint(self.setup_rerun_blueprint(n_frames))

        # setup pytorch3d axis
        ru.pt3d_setup()

        # configure attn processor for rerun
        self.attn_processor.rerun_module_paths = self.rerun_module_paths
        self.attn_processor.rerun = self.rerun
        self.attn_processor.rerun_frame_indices = self.rerun_frame_indices

        # return custom timesequence
        seq = ru.TimeSequence("timesteps")
        return seq

    def log_latents(self, latents, generator=None):

        if self.rerun:

            for f_i in self.rerun_frame_indices:
                latent = latents[f_i]
                rr.log(f"latent_{f_i}", rr.Tensor(rearrange(latent, "c w h -> c h w")))
                cur_img = self.latents_to_images(latent.unsqueeze(0), generator)[0]
                rr.log(f"frame_{f_i}", rr.Image(cur_img))

    @torch.no_grad()
    @typechecked
    def __call__(
        self,
        # inputs
        prompt: str,
        frames: Meshes,
        cameras: FoVPerspectiveCameras,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        # config
        res: int,
        num_inference_steps: int,
        do_uv_noise_init: bool,
        uv_texture_res: int,
        do_pre_attn_injection: bool,
        do_post_attn_injection: bool,
        feature_blend_alpha: float,
        mean_features_weight: float,
        num_keyframes: int,
        guidance_scale: float,
        controlnet_conditioning_scale: float,
        chunk_size: int,
        module_paths: List[str],
        seed: int,
        # logging config
        rerun: bool,
        rerun_module_paths: List[str],
        save_tensors: bool,
    ):

        self.update_params(
            save_tensors=save_tensors,
            rerun_module_paths=rerun_module_paths,
            rerun=rerun,
            res=res,
            num_inference_steps=num_inference_steps,
            do_uv_noise_init=do_uv_noise_init,
            uv_texture_res=uv_texture_res,
            do_pre_attn_injection=do_pre_attn_injection,
            do_post_attn_injection=do_post_attn_injection,
            feature_blend_alpha=feature_blend_alpha,
            num_keyframes=num_keyframes,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            chunk_size=chunk_size,
            module_paths=module_paths,
            seed=seed,
            mean_features_weight=mean_features_weight,
        )

        if self.save_tensors:
            tensors = TensorsArtifact.create_empty_artifact("tensors")
            tensors.open_h5_file()

        n_frames = len(frames)

        # setup rerun
        rerun_seq = self.setup_rerun(n_frames)

        # setup generator
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

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
        if self.do_uv_noise_init:
            latents = self.prepare_uv_initialized_latents(
                frames, cameras, verts_uvs, faces_uvs, generator=generator
            )
        else:
            latents = self.prepare_random_latents(n_frames, generator=generator)

        # chunk indices to use in inference loop
        chunks_indices = torch.split(torch.arange(0, n_frames), chunk_size)

        # get 2D vertex positions for each frame
        vert_xys, vert_indices = project_vertices_to_cameras(frames, cameras)

        # denoising loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):

            self.log_latents(latents, generator)

            # duplicate latent, to feed to model with CFG
            latents_stacked = torch.stack([latents] * 2)
            latents_stacked = self.scheduler.scale_model_input(latents_stacked, t)

            # sample keyframe indices
            kf_indices = self.sample_keyframes(n_frames)

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

            # Unify features across keyframes as vertex features
            kf_vert_xys = [vert_xys[i] for i in kf_indices.tolist()]
            kf_vert_indices = [vert_indices[i] for i in kf_indices.tolist()]
            aggregated_3d_features = self.aggregate_kf_features(
                frames.num_verts_per_mesh()[0],
                kf_vert_xys,
                kf_vert_indices,
                post_attn_features,
            )

            if self.rerun:
                rr_module_path = self.rerun_module_paths[0]
                rr_vert_features = aggregated_3d_features[rr_module_path][0][0]
                pca = RgbPcaUtil.init_from_features(rr_vert_features.cpu())
                self.attn_processor.pca = pca
                rr_vert_features_rgb = pca.features_to_rgb(rr_vert_features.cpu())

                rr.log(
                    "mesh",
                    ru.pt3d_mesh(frames[0], vertex_colors=rr_vert_features_rgb),
                )

            # do inference in chunks
            noise_preds = []

            for chunk_indices in tqdm(chunks_indices, desc="Chunks"):

                # get chunk inputs
                chunk_latents = latents_stacked[:, chunk_indices]
                chunk_embeddings = stacked_text_embeddings[:, chunk_indices]
                chunk_depth_maps = [depth_maps[i] for i in chunk_indices.tolist()]

                # render chunk feature images
                chunk_feature_images = self.render_feature_images(
                    cameras[chunk_indices],
                    frames[chunk_indices],
                    aggregated_3d_features,
                )

                # Diffusion step #2 with pre and post attn feature injection
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
            noise_pred = noise_pred_uncond + guidance_scale * guidance_direction

            # update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            rerun_seq.step()

        self.log_latents(latents, generator)

        # decode latents in chunks
        decoded_imgs = []
        for chunk_indices in chunks_indices:
            chunk_latents = latents[chunk_indices]
            chunk_images = self.latents_to_images(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        return decoded_imgs
