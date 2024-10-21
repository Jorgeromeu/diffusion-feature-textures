import torch
import torchvision.transforms.functional as TF
from einops import rearrange
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    BlendParams,
    CamerasBase,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from torch import Tensor, nn


class FeatureShader(nn.Module):
    def __init__(self, device="cuda", blend_params=BlendParams()):
        super().__init__()
        self.device = device
        self.blend_params = blend_params

    def forward(self, fragments, meshes):
        # get the vertex features
        texels = meshes.sample_textures(fragments)

        valid_max = fragments.pix_to_face >= 0

        blended_texels = torch.zeros_like(texels)
        blended_texels[valid_max] = texels[valid_max]

        # TODO blending
        return blended_texels[:, :, :, 0, :]


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


def make_rasterizer(cameras, resolution=512):
    raster_settings = RasterizationSettings(image_size=resolution, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    return rasterizer


def render_depth_map(meshes, cameras, resolution=512):
    rasterizer = make_rasterizer(cameras, resolution)
    fragments = rasterizer(meshes)
    depth_maps = fragments.zbuf
    depth_maps = normalize_depth_map(depth_maps)
    return [TF.to_pil_image(depth_map[:, :, 0]) for depth_map in depth_maps]


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


def rasterize_vertex_features(
    cameras: CamerasBase, meshes: Meshes, res: int, vertex_features: Tensor
):
    rasterizer = make_rasterizer(cameras, res)

    renders = []
    for mesh in meshes:
        fragments = rasterizer(mesh)

        # B, F, V, D storing feature for each vertex in each face
        face_vert_features = vertex_features[mesh.faces_list()[0]]

        # interpolate with barycentric coords
        pixel_features = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, face_vert_features
        )

        pixel_features = rearrange(pixel_features, "1 h w 1 d -> d h w")
        renders.append(pixel_features)

    return torch.stack(renders)
