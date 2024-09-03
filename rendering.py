from pytorch3d.renderer import (
    RasterizationSettings, MeshRasterizer,
    AmbientLights, SoftPhongShader, MeshRenderer
)
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes

def process_depth_map(depth):
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
    raster_settings = RasterizationSettings(image_size=resolution, faces_per_pixel=1)
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

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

