import torch
from einops import rearrange
from pytorch3d.renderer import (
    CamerasBase,
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from torch import Tensor

from text3d2video.backprojection import make_repeated_uv_texture
from text3d2video.rendering import TextureShader


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
        # sample noise uv_map
        noise_uv_map = torch.randn(
            self.noise_texture_res,
            self.noise_texture_res,
            self.noise_channels,
            device=device,
            generator=generator,
        )

        # create noisy texture
        n_frames = len(meshes)
        noise_texture = make_repeated_uv_texture(
            noise_uv_map, faces_uvs, verts_uvs, n_frames
        )
        noise_texture.sampling_mode = sampling_mode
        meshes.textures = noise_texture

        # rasterize
        raster_settings = RasterizationSettings(
            image_size=64,
            faces_per_pixel=1,
            bin_size=0,
        )
        rasterizer = MeshRasterizer(raster_settings=raster_settings)
        fragments = rasterizer(meshes, cameras=cameras)

        # render
        shader = TextureShader()
        noise_renders = shader(fragments, meshes)
        noise_renders = noise_renders.to(device=device, dtype=dtype)

        # sample background noise
        bg_noise = torch.randn(
            self.noise_channels,
            noise_resolution,
            noise_resolution,
            generator=generator,
            device=device,
            dtype=dtype,
        )

        # background for each frame
        background_noise = bg_noise.expand(n_frames, -1, -1, -1)

        masks = fragments.pix_to_face > 0
        masks = rearrange(masks, "N H W 1 -> N 1 H W")

        return ~masks * background_noise + noise_renders
