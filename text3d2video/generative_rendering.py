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
from text3d2video.rendering import render_depth_map
from text3d2video.sd_feature_extraction import SAFeatureExtractor
from text3d2video.util import front_camera
from text3d2video.sd_feature_extraction import get_module_from_path
import rerun as rr
import text3d2video.rerun_util as ru
import rerun.blueprint as rrb


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

        return rrb.Blueprint(
            rrb.Tabs(main_tab),
            collapse_panels=True,
        )

    @torch.no_grad()
    @typechecked
    def __call__(
        self,
        prompt: str,
        frames: Meshes,
        res=512,
        num_inference_steps=30,
        guidance_scale=7.5,
        controlnet_conditioning_scale=1.0,
        generator=None,
        rerun=False,
    ):

        # setup rerun
        ru.set_logging_state(rerun)
        rr.init("generative_rendering")
        rr.serve()
        rr.send_blueprint(self.setup_blueprint(len(frames)))
        seq = ru.TimeSequence("timesteps")

        # number of images being generated
        batch_size = len(frames)
        prompts = [prompt] * batch_size

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

        # initialize latents from standard normal
        latents = self.prepare_latents(batch_size, res, generator)

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

            # controlnet step
            controlnet_model_input = latent_model_input
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

            # diffusion step to extract features
            extractor = SAFeatureExtractor()

            for attn_path in self.module_paths:
                attn = get_module_from_path(self.unet, attn_path)
                extractor.add_attn_hooks(attn, attn_path)

            self.unet(
                latent_model_input,
                t,
                mid_block_additional_residual=mid_block_res_sample,
                down_block_additional_residuals=down_block_res_samples,
                encoder_hidden_states=text_embeddings,
            )

            extractor.hooks.clear_all_hooks()

            # diffusion step with:
            # - controlnet residuals
            # - classifier free guidance
            # - TODO pre attn feature injection
            # - TODO post attn feature injection

            self.unet.do_pre_attn_injection = True
            attn_processor.saved_pre_attn = extractor.saved_inputs
            attn_processor.savd_outputs = extractor.saved_outputs

            noise_pred = self.unet(
                latent_model_input,
                t,
                mid_block_additional_residual=mid_block_res_sample,
                down_block_additional_residuals=down_block_res_samples,
                encoder_hidden_states=text_embeddings,
            ).sample

            # preform classifier free guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            guidance_direction = noise_pred_text - noise_pred_uncond
            noise_pred = noise_pred_uncond + guidance_scale * guidance_direction

            # update latents
            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            seq.step()

        # decode latents
        return self.latents_to_images(latents, generator)
