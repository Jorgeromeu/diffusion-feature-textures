import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from jaxtyping import Float
from pytorch3d.renderer import (
    MeshRasterizer,
    RasterizationSettings,
    TexturesUV,
    TexturesVertex,
)
from pytorch3d.structures import Meshes
from torch import Tensor, nn


class TextureShader(nn.Module):
    """
    Simple shader, that returns textured colors of mesh, no shading
    """

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

    def forward(self, fragments, meshes: Meshes, **kwargs):
        colors = meshes.sample_textures(fragments)
        mask = fragments.pix_to_face > 0
        output = torch.zeros_like(colors)
        output[mask] = colors[mask]
        output = output[:, :, :, 0, :]
        output = rearrange(output, "b h w c -> b c h w")
        return output


def normalize_depth_map(depth):
    """
    Convert from zbuf to depth map
    """

    max_depth = depth.max()
    indices = depth == -1
    depth = max_depth - depth
    depth[indices] = 0
    max_depth = depth.max()
    depth = depth / max_depth
    return depth


def make_mesh_rasterizer(
    cameras=None,
    resolution=512,
    faces_per_pixel=1,
    blur_radius=0,
    bin_size=0,
):
    raster_settings = RasterizationSettings(
        image_size=resolution,
        faces_per_pixel=faces_per_pixel,
        blur_radius=blur_radius,
        bin_size=bin_size,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    return rasterizer


def render_depth_map(meshes, cameras, resolution=512, chunk_size=30):
    rasterizer = make_mesh_rasterizer(resolution=resolution)
    indices = torch.arange(0, len(meshes))

    all_depth_maps = []
    for chunk_indices in torch.split(indices, chunk_size):
        chunk_meshes = meshes[chunk_indices]
        chunk_cameras = cameras[chunk_indices]
        fragments = rasterizer(chunk_meshes, cameras=chunk_cameras)
        depth_maps = fragments.zbuf
        depth_maps = normalize_depth_map(depth_maps)
        depth_maps = [TF.to_pil_image(depth_map[:, :, 0]) for depth_map in depth_maps]
        all_depth_maps.extend(depth_maps)

    return all_depth_maps


def make_repeated_vert_texture(vert_features: Float[Tensor, "n c"], N=1):
    extended_vt_features = vert_features.unsqueeze(0).expand(N, -1, -1)
    return TexturesVertex(extended_vt_features)


def make_repeated_uv_texture(
    uv_map: Float[Tensor, "h w c"], faces_uvs: Tensor, verts_uvs: Tensor, N=1
):
    extended_uv_map = uv_map.to(torch.float32)  # pt3d requires float32 for textures
    extended_uv_map = extended_uv_map.unsqueeze(0).expand(N, -1, -1, -1)
    extended_faces_uvs = faces_uvs.unsqueeze(0).expand(N, -1, -1)
    extended_verts_uvs = verts_uvs.unsqueeze(0).expand(N, -1, -1)
    return TexturesUV(extended_uv_map, extended_faces_uvs, extended_verts_uvs)
