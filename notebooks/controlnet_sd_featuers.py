# %% Imports
import torchvision.transforms.functional as TF
from sd_feature_extraction import SDFeatureExtractor
from diffusers import UNet2DConditionModel
from text3d2video.rendering import rasterize, normalize_depth_map
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
from text3d2video.obj_io import OBJAnimation
from diffusion import depth2img_pipe, depth2img
from visualization import reduce_feature_map
import matplotlib.pyplot as plt
import torch
from text3d2video.feature_pipeline import FeaturePipeline
from einops import rearrange
import numpy as np
import sys
from pathlib import Path
from IPython import get_ipython

ipy = get_ipython()
ipy.extension_manager.load_extension('autoreload')
ipy.run_line_magic('autoreload', '2')

sys.path.append('../')


# %% Get depth map
device = torch.device("cuda:0")

animation_path = Path('../data/backflip')
animation = OBJAnimation(animation_path)
mesh = animation.load_frame(1)

R, T = look_at_view_transform(dist=2, azim=0, elev=0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60)
# %%
res = 512
fragments, depth_map = rasterize(cameras, mesh, res)
depth_map_normalized = normalize_depth_map(depth_map).to(device).unsqueeze(0)

depth_img = TF.to_pil_image(depth_map_normalized)

# %% Load Pipeline
device = 'cuda:0'
pipe = depth2img_pipe()

# %% UNet

pipe.unet

# %% Generate Image
feature_extractor = SDFeatureExtractor(pipe)
img_out = depth2img(pipe, 'Deadpool Dancing', depth_img)
img_out

# %% inspect features
feature = feature_extractor.get_feature(level=2, timestep=0)


feature_pca = reduce_feature_map(feature)
print(feature_pca.max())

plt.axis('off')
plt.imshow(feature_pca.permute(1, 2, 0))
