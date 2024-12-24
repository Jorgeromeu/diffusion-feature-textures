import torch
from einops import rearrange
from pytorch3d.renderer import CamerasBase, TexturesUV
from pytorch3d.structures import Meshes
from torch import Tensor

from text3d2video.rendering import make_feature_renderer


class NoiseInitializer:
    """
    Base class for noise initialization methods
    """

    def __init__(self, noise_channels: int = 4, noise_resolution: int = 64):
        self.noise_channels = noise_channels
        self.noise_resolution = noise_resolution

    def initial_noise(
        self, device="cuda", dtype=torch.float16, generator=None, **kwargs
    ) -> Tensor:
        pass


class RandomNoiseInitializer(NoiseInitializer):
    def initial_noise(
        self,
        n_frames: int,
        device="cuda",
        dtype=torch.float16,
        generator=None,
        **kwargs,
    ):
        return torch.randn(
            n_frames,
            self.noise_channels,
            self.noise_resolution,
            self.noise_resolution,
            generator=generator,
            device=device,
            dtype=dtype,
        )


class FixedNoiseInitializer(NoiseInitializer):
    def initial_noise(
        self,
        n_frames: int,
        device="cuda",
        dtype=torch.float16,
        generator=None,
        **kwargs,
    ):
        noise_0 = torch.randn(
            self.noise_channels,
            self.noise_resolution,
            self.noise_resolution,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        return noise_0.expand(n_frames, -1, -1, -1)


class UVNoiseInitializer(NoiseInitializer):
    noise_texture_res: int

    def __init__(self, noise_channels=4, noise_resolution=64, noise_texture_res=64):
        super().__init__(noise_channels, noise_resolution)
        self.noise_texture_res = noise_texture_res

    def initial_noise(
        self,
        meshes: Meshes,
        cameras: CamerasBase,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        generator=None,
        noise_resolution=64,
        sampling_mode: str = "nearest",
        device="cuda",
        dtype=torch.float16,
        **kwargs,
    ):
        # setup noise texture
        noise_texture_map = torch.randn(
            self.noise_texture_res,
            self.noise_texture_res,
            self.noise_channels,
            device=device,
            generator=generator,
        )

        # create noise texture for meshes
        n_frames = len(meshes)
        noise_texture = TexturesUV(
            verts_uvs=verts_uvs.expand(n_frames, -1, -1).to(device),
            faces_uvs=faces_uvs.expand(n_frames, -1, -1).to(device),
            maps=noise_texture_map.expand(n_frames, -1, -1, -1).to(device),
        )
        noise_texture.sampling_mode = sampling_mode
        meshes.textures = noise_texture

        # render noise texture for each frame
        renderer = make_feature_renderer(cameras, noise_resolution)
        noise_renders = renderer(meshes).to(dtype)

        noise_renders = rearrange(noise_renders, "b h w c -> b c h w")
        noise_renders = noise_renders.to(device=device, dtype=dtype)

        # create consistent noise for background
        background_noise = torch.randn(
            self.noise_channels,
            noise_resolution,
            noise_resolution,
            generator=generator,
            device=device,
        ).expand(n_frames, -1, -1, -1)
        background_noise = background_noise.to(device, dtype=dtype)

        latents_mask = (noise_renders == 0).float()
        latents = noise_renders + background_noise * latents_mask

        latents = latents.to(device, dtype=dtype)

        return latents
