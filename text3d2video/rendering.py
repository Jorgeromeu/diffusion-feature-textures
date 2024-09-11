from einops import rearrange
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    RasterizationSettings, MeshRasterizer,
    AmbientLights, SoftPhongShader, MeshRenderer
)
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes

EXTENT_UV = [0, 1, 0, 1]

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


def init_renderer(cameras, device='cuda:0', resolution=512):
    raster_settings = RasterizationSettings(
        image_size=resolution, faces_per_pixel=1)
    rasterizer = MeshRasterizer(
        cameras=cameras, raster_settings=raster_settings)

    lights = AmbientLights(device=device)

    shader = SoftPhongShader(
        cameras=cameras,
        lights=lights,
        device=device
    )

    renderer = MeshRenderer(
        rasterizer=rasterizer,
        shader=shader
    )

    return renderer


def rasterize_mesh(renderer: MeshRenderer, meshes: Meshes) -> Fragments:
    fragments: Fragments = renderer.rasterizer(meshes)

    # extract zbuf
    zbuf = fragments.zbuf[0, :, :, 0]

    return fragments, zbuf


def rasterize(cameras, mesh, res=100):
    renderer = init_renderer(cameras, resolution=res)
    fragments, depth_map = rasterize_mesh(renderer, mesh)
    return fragments, depth_map


def rasterize_vertex_features(cameras, mesh, res, vertex_features):

    # rasterize mesh from camera
    fragments, _ = rasterize(cameras, mesh, res)

    # F, V, D storing feature for each vertex in each face
    face_vert_features = vertex_features[mesh.faces_list()[0]]

    # interpolate with barycentric coords
    pixel_features = interpolate_face_attributes(
        fragments.pix_to_face,
        fragments.bary_coords,
        face_vert_features
    )

    pixel_features = rearrange(pixel_features, '1 h w 1 d -> d h w')

    return pixel_features
