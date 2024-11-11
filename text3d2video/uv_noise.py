import torch
from einops import rearrange, repeat
from jaxtyping import Float
from pytorch3d.renderer import FoVPerspectiveCameras, TexturesUV
from pytorch3d.structures import Meshes
from torch import Tensor

from text3d2video.generative_rendering.configs import (
    NoiseInitializationConfig,
    NoiseInitializationMethod,
)
from text3d2video.rendering import make_feature_renderer


def prepare_uv_initialized_latents(
    meshes: Meshes,
    cameras: FoVPerspectiveCameras,
    verts_uvs: torch.Tensor,
    faces_uvs: torch.Tensor,
    generator=None,
    latent_res=64,
    latent_texture_res=64,
    latent_channels=4,
    sampling_mode: str = "nearest",
    device="cuda",
    dtype=torch.float16,
) -> Float[Tensor, "b c w h"]:
    # setup noise texture
    noise_texture_map = torch.randn(
        latent_texture_res,
        latent_texture_res,
        latent_channels,
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
    renderer = make_feature_renderer(cameras, latent_res)
    noise_renders = renderer(meshes).to(dtype)

    noise_renders = rearrange(noise_renders, "b h w c -> b c h w")
    noise_renders = noise_renders.to(device=device, dtype=dtype)

    # create consistent noise for background
    background_noise = torch.randn(
        latent_channels, latent_res, latent_res, generator=generator, device=device
    ).expand(n_frames, -1, -1, -1)
    background_noise = background_noise.to(device, dtype=dtype)

    latents_mask = (noise_renders == 0).float()
    latents = noise_renders + background_noise * latents_mask

    latents = latents.to(device, dtype=dtype)

    return latents


def prepare_latents(
    meshes: Meshes,
    cameras: FoVPerspectiveCameras,
    verts_uvs,
    faces_uvs,
    noise_initialization_config: NoiseInitializationConfig,
    latent_resolution=64,
    generator=None,
    device="cuda",
    latent_channels=4,
    dtype=torch.float16,
):
    n_frames = len(meshes)

    # type of noise initialization
    noise_init_method = noise_initialization_config.method

    # UV noise
    if noise_init_method == NoiseInitializationMethod.UV:
        return prepare_uv_initialized_latents(
            meshes,
            cameras,
            verts_uvs,
            faces_uvs,
            generator,
            latent_channels=latent_channels,
            latent_res=latent_resolution,
            latent_texture_res=noise_initialization_config.uv_texture_res,
            device=device,
            dtype=dtype,
        )

    # fully random noise
    if noise_init_method == NoiseInitializationMethod.RANDOM:
        return torch.randn(
            n_frames,
            latent_channels,
            latent_resolution,
            latent_resolution,
            generator=generator,
            device=device,
            dtype=dtype,
        )

    # fixed noise across frames
    if noise_init_method == NoiseInitializationMethod.FIXED:
        latent_0 = torch.randn(
            latent_channels,
            latent_resolution,
            latent_resolution,
            generator=generator,
            device=device,
            dtype=dtype,
        )
        return repeat(latent_0, "c h w -> b c h w", b=n_frames)

    raise ValueError(f"Invalid noise initialization method: {noise_init_method}")
