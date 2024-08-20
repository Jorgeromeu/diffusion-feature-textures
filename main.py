import matplotlib.pyplot as plt
import torch
import warnings
from pytorch3d.io import load_obj
from pytorch3d.renderer import (
    MeshRenderer, MeshRasterizer, HardPhongShader,
    look_at_view_transform, FoVPerspectiveCameras, RasterizationSettings, SoftSilhouetteShader
)
from pytorch3d.structures import Meshes

device = torch.device("cuda:0")

verts, faces, aux = load_obj('data/spot.obj', device=device)

meshes = Meshes(verts=[verts], faces=[faces.verts_idx])
# Initialize an OpenGL perspective camera.
R, T = look_at_view_transform(2.7, 10, 20)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T)

raster_settings = RasterizationSettings(
    image_size=512,
    blur_radius=0.0,
    faces_per_pixel=1,
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftSilhouetteShader()
)

images = renderer(meshes, cameras=cameras)

im = images[0, ..., 2].cpu()

print(im.shape)

plt.imshow(im)
plt.show()
