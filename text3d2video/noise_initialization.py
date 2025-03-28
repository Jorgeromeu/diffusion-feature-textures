import torch
from einops import rearrange
from pytorch3d.renderer import (
    CamerasBase,
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from torch import Tensor

from text3d2video.rendering import TextureShader, make_repeated_uv_texture


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
    uv_noise: Tensor
    bg_noise: Tensor

    def __init__(
        self,
        noise_channels=4,
        noise_resolution=64,
        noise_texture_res=64,
        include_background=True,
        device="cuda",
        dtype=torch.float16,
    ):
        super().__init__(noise_channels, noise_resolution)
        self.dtype = dtype
        self.device = device
        self.noise_texture_res = noise_texture_res
        self.include_background = include_background

        self.uv_noise = None
        self.bg_noise = None

    @classmethod
    def init_from_textures(cls, uv_noise: Tensor, bg_noise: Tensor):
        noise_texture_res, _, noise_channels = uv_noise.shape
        noise_resolution = bg_noise.shape[-1]
        device = uv_noise.device
        dtype = uv_noise.dtype
        instance = cls(
            noise_channels=noise_channels,
            noise_resolution=noise_resolution,
            noise_texture_res=noise_texture_res,
            device=device,
            dtype=dtype,
        )
        instance.uv_noise = uv_noise
        instance.bg_noise = bg_noise
        return instance

    def sample_noise_texture(self, generator=None):
        self.uv_noise = torch.randn(
            self.noise_texture_res,
            self.noise_texture_res,
            self.noise_channels,
            device=self.device,
            generator=generator,
        )

    def sample_background(self, generator=None):
        self.bg_noise = torch.randn(
            self.noise_channels,
            self.noise_resolution,
            self.noise_resolution,
            generator=generator,
            device=self.device,
            dtype=self.dtype,
        )

    def initial_noise(
        self,
        meshes: Meshes,
        cameras: CamerasBase,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        sampling_mode: str = "nearest",
        generator=None,
        **kwargs,
    ):
        # sample texture
        if self.uv_noise is None:
            self.sample_noise_texture(generator)

        # setup meshes
        n_frames = len(meshes)
        noise_texture = make_repeated_uv_texture(
            self.uv_noise,
            faces_uvs,
            verts_uvs,
            n_frames,
            sampling_mode=sampling_mode,
        )
        meshes.textures = noise_texture

        # rasterize
        raster_settings = RasterizationSettings(
            image_size=self.noise_resolution,
            faces_per_pixel=1,
            bin_size=0,
        )
        rasterizer = MeshRasterizer(raster_settings=raster_settings)
        fragments = rasterizer(meshes, cameras=cameras)

        # shade
        shader = TextureShader()
        noise_renders = shader(fragments, meshes)
        noise_renders = noise_renders.to(self.device, self.dtype)

        if not self.include_background:
            return noise_renders

        # sample background noise
        if self.bg_noise is None:
            self.sample_background(generator)

        # background for each frame
        background_noise = self.bg_noise.expand(n_frames, -1, -1, -1)

        masks = fragments.pix_to_face > 0
        masks = rearrange(masks, "N H W 1 -> N 1 H W")

        return ~masks * background_noise + noise_renders
