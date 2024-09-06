# %% Imports
from IPython import get_ipython
from einops import rearrange

ipy = get_ipython()
ipy.extension_manager.load_extension('autoreload')
ipy.run_line_magic('autoreload', '2')

import sys
sys.path.append('../')

import matplotlib.pyplot as plt
from PIL import Image
import torch
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
import torchvision.transforms as transforms
from file_util import load_frame_obj
from pathlib import Path
from rendering import normalize_depth_map, rasterize
from util import feature_per_vertex, sample_feature_map
from diffusion import depth2img_pipe, depth2img

to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

# %%
from file_util import OBJAnimation

device = torch.device("cuda:0")

animation_path = Path('../data/backflip')
animation = OBJAnimation(animation_path)
mesh = animation.load_frame(1)

# %%
R, T = look_at_view_transform(dist=2, azim=0, elev=0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60)

# %%
res = 512
fragments, depth_map = rasterize(cameras, mesh, res)

depth_map_normalized = normalize_depth_map(depth_map).to(device)
depth_img = to_pil(depth_map_normalized)

depth_img

# %%
pipe = depth2img_pipe()

# %%
img_out = depth2img(pipe, 'Deadpool Dancing', depth_img)
img_out

# %%
feature_tensor = to_tensor(img_out).to(device)
img_out

# %%
vert_features = feature_per_vertex(mesh, cameras, feature_tensor)
face_vert_features = vert_features[mesh.faces_list()[0]]

# %%
from rendering import rasterize_vertex_features

reposed = animation.load_frame(1)
pixel_features = rasterize_vertex_features(cameras, reposed, res, vert_features)

plt.imshow(pixel_features.cpu())
plt.axis('off')

