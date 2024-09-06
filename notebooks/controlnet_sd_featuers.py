"""
"""

# %% Imports
from pathlib import Path
from IPython import get_ipython

ipy = get_ipython()
ipy.extension_manager.load_extension('autoreload')
ipy.run_line_magic('autoreload', '2')

import sys
sys.path.append('../')

from sklearn.preprocessing import MinMaxScaler
import faiss
import numpy as np
from einops import rearrange
from feature_pipeline import FeaturePipeline
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from visualization import reduce_feature_map
from diffusion import depth2img_pipe, depth2img
from file_util import OBJAnimation
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
from rendering import rasterize, normalize_depth_map
from diffusers import UNet2DConditionModel
from sd_feature_extraction import SDFeatureExtractor

# %% Get depth map
device = torch.device("cuda:0")

animation_path = Path('../data/backflip')
animation = OBJAnimation(animation_path)
mesh = animation.load_frame(1)

# %%
R, T = look_at_view_transform(dist=2, azim=0, elev=0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60)

# %%
import torchvision.transforms as transforms
to_pil = transforms.ToPILImage()

res = 512
fragments, depth_map = rasterize(cameras, mesh, res)

depth_map_normalized = normalize_depth_map(depth_map).to(device)
depth_img = to_pil(depth_map_normalized)
depth_img

# %% Load Pipeline
device = 'cuda:0'
pipe = depth2img_pipe()

# %% add hooks
unet: UNet2DConditionModel = pipe.unet
feature_extractor = SDFeatureExtractor(pipe)

# %% Generate Image
img_out = depth2img(pipe, 'Deadpool Dancing', depth_img)
img_out

# %% inspect features
feature = feature_extractor.get_feature(level=2, timestep=0)

from visualization import reduce_feature_map

feature_pca = reduce_feature_map(feature)

plt.imshow(feature_pca.permute(1,2,0).cpu().numpy())