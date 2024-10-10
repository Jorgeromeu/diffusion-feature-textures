from math import sqrt
from typing import List
from diffusers import (
    AutoencoderKL,
    DiffusionPipeline,
    UNet2DConditionModel,
    UniPCMultistepScheduler,
    ControlNetModel,
)
from einops import rearrange
import torch
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from diffusers.image_processor import VaeImageProcessor
from typeguard import typechecked
from pytorch3d.structures import Meshes
from text3d2video.generative_rendering_attn import GenerativeRenderingAttn
from text3d2video.rendering import make_feature_renderer, render_depth_map
from text3d2video.sd_feature_extraction import SAFeatureExtractor
from text3d2video.util import front_camera
from text3d2video.sd_feature_extraction import get_module_from_path
import rerun as rr
import text3d2video.rerun_util as ru
import rerun.blueprint as rrb

from text3d2video.visualization import RgbPcaUtil
from pytorch3d.renderer import TexturesUV


class GenerativeRenderingPipeline(DiffusionPipeline):

    current_step: int = None
    current_step_index: int = None

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

    def encode_prompt(self, prompts: List[str]):

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

    def prepare_latents(self, batch_size: int, out_resolution: int, generator=None):
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
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        out_resolution: int = 512,
        generator=None,
    ):

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
        camera = front_camera()
        renderer = make_feature_renderer(camera, latent_res)
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
        controlnet_conditioning_scale: float = 1.0,
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
            conditioning_scale=controlnet_conditioning_scale,
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

    def setup_blueprint(self, n_frames: int):

        frame_views = []
        latent_views = []
        depth_map_views = []

        for frame_i in range(n_frames):
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

        pose_views = [
            rrb.Spatial3DView(contents=[f"+/frame_{i}"]) for i in range(n_frames)
        ]

        features_3d_tab = rrb.Vertical(
            rrb.Horizontal(*pose_views, name="Poses"),
        )

        return rrb.Blueprint(
            rrb.Tabs(main_tab, features_3d_tab),
            collapse_panels=True,
        )

    @torch.no_grad()
    @typechecked
    def __call__(
        self,
        prompt: str,
        frames: Meshes,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        res=512,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        num_keyframes=2,
        generator=None,
        rerun=False,
    ):

        # setup rerun
        ru.set_logging_state(rerun)
        rr.init("generative_rendering")
        rr.serve()
        rr.send_blueprint(self.setup_blueprint(len(frames)))
        ru.pt3d_setup()
        seq = ru.TimeSequence("timesteps")

        # number of images being generated
        n_frames = len(frames)
        prompts = [prompt] * n_frames

        # render depth maps
        camera = front_camera()
        depth_maps = render_depth_map(frames, camera, res)

        for i, depth_map in enumerate(depth_maps):
            rr.log(f"depth_map_{i}", rr.Image(depth_map))

        # Get prompt embeddings for guidance
        cond_embeddings, uncond_embeddings = self.encode_prompt(prompts)
        text_embeddings = torch.cat([uncond_embeddings, cond_embeddings])

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)

        # uv-initialized latents
        latents = self.prepare_uv_initialized_latents(frames, verts_uvs, faces_uvs)

        # setup attn processor
        attn_processor = GenerativeRenderingAttn(self.unet, unet_chunk_size=2)
        self.unet.set_attn_processor(attn_processor)

        # denoising loop
        for i, t in enumerate(tqdm(self.scheduler.timesteps)):

            if rerun:
                for f_i, latent in enumerate(latents):
                    rr.log(
                        f"latent_{f_i}", rr.Tensor(rearrange(latent, "c w h -> c h w"))
                    )

                cur_images = self.latents_to_images(latents, generator)
                for f_i, im in enumerate(cur_images):
                    rr.log(f"frame_{f_i}", rr.Image(im))

            # duplicate latent, to feed to model with CFG
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            latents_kf = rearrange(
                latent_model_input, "(b f) h w c -> b f h w c", f=n_frames
            )
            kf_indices = torch.randperm(n_frames)[:num_keyframes]
            latents_kf = latents_kf[:, kf_indices]
            latents_kf = rearrange(latents_kf, "b f h w c -> (b f) h w c")

            text_embeddings_kf = rearrange(
                text_embeddings, "(b f) t d -> b f t d", f=n_frames
            )
            text_embeddings_kf = text_embeddings_kf[:, kf_indices]
            text_embeddings_kf = rearrange(text_embeddings_kf, "b f t d -> (b f) t d")

            attn_processor.do_extended_attention = True
            self.controlnet_and_unet_forward(
                latents_kf,
                text_embeddings_kf,
                [depth_maps[i] for i in kf_indices.tolist()],
                t,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
            )
            attn_processor.do_extended_attention = False

            # diffusion step on all with:
            # - controlnet residuals
            # - classifier free guidance
            # - TODO pre attn feature injection
            # - TODO post attn feature injection

            # pass extracted features to attn processor
            attn_processor.do_pre_attn_injection = True
            attn_processor.do_post_attn_injection = False
            noise_pred = self.controlnet_and_unet_forward(
                latent_model_input,
                text_embeddings,
                depth_maps,
                t,
                controlnet_conditioning_scale=controlnet_conditioning_scale,
            ).sample
            attn_processor.do_pre_attn_injection = False
            attn_processor.do_post_attn_injection = False

            # preform classifier free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guidance_direction = noise_pred_text - noise_pred_uncond
            noise_pred = noise_pred_uncond + guidance_scale * guidance_direction
            noise_pred = noise_pred_uncond

            # update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample
            seq.step()

        # decode latents
        return self.latents_to_images(latents, generator)
