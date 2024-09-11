# %% Imports
ipy = get_ipython()
ipy.extension_manager.load_extension('autoreload')
ipy.run_line_magic('autoreload', '2')

import sys
sys.path.append('../')

from IPython import get_ipython
from einops import einsum, rearrange
from tqdm import tqdm
import sd_feature_extraction
from visualization import reduce_feature_map
from sd_feature_extraction import SDFeatureExtractor
import matplotlib.pyplot as plt
from PIL import Image
import torch
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
import torchvision.transforms as transforms
from text3d2video.file_util import load_frame_obj
from pathlib import Path
from text3d2video.rendering import normalize_depth_map, rasterize
from util import feature_per_vertex, sample_feature_map
from diffusion import depth2img_pipe, depth2img

to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

# %%
from text3d2video.file_util import OBJAnimation

device = torch.device("cuda:0")

animation_path = Path('../data/dancing')
animation = OBJAnimation(animation_path)
mesh = animation.load_frame(1)

# %%
R, T = look_at_view_transform(dist=2, azim=0, elev=0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60)

res = 512
fragments, depth_map = rasterize(cameras, mesh, res)
depth_map_normalized = normalize_depth_map(depth_map).to(device)
depth_img = to_pil(depth_map_normalized)
depth_img

# %%
pipe = depth2img_pipe()

# %%
feature_extractor = SDFeatureExtractor(pipe)
img_out = depth2img(pipe, 'Deadpool, blank background', depth_img)
img_out

# %%
feature_map = feature_extractor.get_feature(level=2, timestep=-1)
feature_map = torch.Tensor(feature_map)

feature_map_rgb = reduce_feature_map(feature_map)
# feature_map_rgb = to_tensor(img_out)

_, feature_res, _ = feature_map.shape

plt.imshow(feature_map_rgb.permute(1,2,0).cpu().numpy())

# %%
vert_features = feature_per_vertex(mesh, cameras, feature_map_rgb)


# %%
frames = []
feature_res = feature_map_rgb.shape[1]
for frame_i in tqdm(animation.framenums(sample_n=None)):
    mesh = animation.load_frame(frame_i)
    rendered_features = rasterize_vertex_features(cameras, mesh, feature_res, vert_features)
    frames.append(rendered_features.cpu())

import imageio
with imageio.get_writer(Path('../outs/deadpool.gif'), mode='I', loop=0) as writer:
    for frame in frames:
        frame_pil = to_pil(frame)
        writer.append_data(frame_pil)