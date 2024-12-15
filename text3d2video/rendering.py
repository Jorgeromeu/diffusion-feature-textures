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


class UVShader(nn.Module):
    def __init__(self, device="cuda", blend_params=BlendParams()):
        super().__init__()
        self.device = device
        self.blend_params = blend_params

    def forward(self, fragments, meshes):
        packing_list = [
            i[j]
            for i, j in zip(
                meshes.textures.verts_uvs_list(), meshes.textures.faces_uvs_list()
            )
        ]
        faces_verts_uvs = torch.cat(packing_list)

        pixel_uvs = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
        )

        return pixel_uvs


class FeatureShader(nn.Module):
    def __init__(self, device="cuda", blend_params=BlendParams()):
        super().__init__()
        self.device = device
        self.blend_params = blend_params

    def forward(self, fragments, meshes: Meshes, **kwargs):
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


def make_rasterizer(cameras=None, resolution=512):
    raster_settings = RasterizationSettings(image_size=resolution, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    return rasterizer


def render_depth_map(meshes, cameras, resolution=512, chunk_size=30):
    rasterizer = make_rasterizer(resolution=resolution)
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
