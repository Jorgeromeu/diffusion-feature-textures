import unittest

import torch
import torchvision.transforms.functional as TF
from diffusers.models import ControlNetModel

from text3d2video.artifacts.animation_artifact import AnimationArtifact
from text3d2video.generative_rendering.generative_rendering_pipeline import (
    GenerativeRenderingPipeline,
)


class TestGenerativeRendering(unittest.TestCase):
    def setUp(self):
        device = torch.device("cuda")
        dtype = torch.float16
        sd_repo = "runwayml/stable-diffusion-v1-5"
        controlnet_repo = "lllyasviel/control_v11f1p_sd15_depth"

        controlnet = ControlNetModel.from_pretrained(
            controlnet_repo, torch_dtype=torch.float16
        ).to(device)

        self.pipe: GenerativeRenderingPipeline = (
            GenerativeRenderingPipeline.from_pretrained(
                sd_repo, controlnet=controlnet, torch_dtype=dtype
            ).to(device)
        )

        self.animation = AnimationArtifact.from_wandb_artifact_tag(
            "backflip:latest", download=False
        )

        return super().setUp()

    @torch.no_grad()
    def test_encode_prompts(self):
        prompt = "lol"
        n_frames = 3
        cond_emb, uncond_emb = self.pipe.encode_prompt([prompt] * n_frames)
        emb_stacked = torch.stack([cond_emb, uncond_emb])

        n_tokens = emb_stacked.shape[2]
        token_dim = emb_stacked.shape[3]
        self.assertEqual(emb_stacked.shape, (2, n_frames, n_tokens, token_dim))

    @torch.no_grad()
    def test_prepare_latents(self):
        n_frames = 3
        frame_nums = self.animation.frame_nums(n_frames)
        frames = self.animation.load_frames(frame_nums)
        cameras = self.animation.cameras(frame_nums)
        verts_uv, faces_uv = self.animation.texture_data()
        latents_random = self.pipe.prepare_random_latents(len(frame_nums))
        latents_uv = self.pipe.prepare_uv_initialized_latents(
            frames, cameras, verts_uv, faces_uv
        )

        self.assertEqual(latents_random.shape, latents_uv.shape)

    @torch.no_grad()
    def test_model_forward(self):
        n_frames = 3
        n_batches = 2
        prompt = "lol"

        latents = self.pipe.prepare_random_latents(n_frames)
        cond_emb, uncond_emb = self.pipe.encode_prompt([prompt] * n_frames)

        # stack
        latents_stacked = torch.stack([latents] * n_batches)
        embeddings = torch.stack([cond_emb, uncond_emb])

        # random depth maps
        depth_maps = [
            TF.to_pil_image(torch.zeros(3, 512, 512)) for _ in range(n_frames)
        ]

        # timestep
        t = 0

        with torch.no_grad():
            noise_pred = self.pipe.model_forward(
                latents_stacked, embeddings, depth_maps=depth_maps, t=t
            )

        print(noise_pred.shape)
