import torch
import torchvision.transforms.functional as TF
from pytorch3d.renderer import (
    BlendParams,
    CamerasBase,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from torch import nn


class FeatureShader(nn.Module):
    """
    Simple shader that returns the texture features as the output, no shading
    """

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

    def forward(self, fragments, meshes: Meshes, **kwargs):
        colors = meshes.sample_textures(fragments)
        return colors[:, :, :, 0, :]


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


def make_mesh_rasterizer(cameras=None, resolution=512):
    raster_settings = RasterizationSettings(
        image_size=resolution, faces_per_pixel=1, blur_radius=0
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


def make_feature_renderer(cameras: CamerasBase, resolution: int, device="cuda"):
    # create a rasterizer
    raster_settings = RasterizationSettings(
        image_size=resolution, blur_radius=0.0, faces_per_pixel=3, bin_size=0
    )

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(
        device
    )

    shader = FeatureShader()
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    return renderer
