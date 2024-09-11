# %% Imports
ipy = get_ipython()
ipy.extension_manager.load_extension('autoreload')
ipy.run_line_magic('autoreload', '2')

from IPython import get_ipython
from einops import einsum, rearrange
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torch
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
import torchvision.transforms as transforms
from text3d2video.file_util import load_frame_obj
from pathlib import Path
from text3d2video.rendering import normalize_depth_map, rasterize, rasterize_vertex_features
import torchvision.transforms.functional as TF

# %%
from text3d2video.file_util import OBJAnimation
from text3d2video.visualization import reduce_features

device = torch.device("cuda:0")

animation_path = Path('data/backflip')
animation = OBJAnimation(animation_path)
mesh = animation.load_frame(1)

# %%
R, T = look_at_view_transform(dist=2, azim=0, elev=0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60)

vert_features = torch.load('outs/mixamo-human_vert_features.pt')
vert_features = reduce_features(vert_features.cpu())
vert_features = torch.Tensor(vert_features).to(device)

# %%
mesh = animation.load_frame(10)
rendered_features = rasterize_vertex_features(cameras, mesh, 100, vert_features)

# %%
frames = []
feature_res = 100
for frame_i in tqdm(animation.framenums(sample_n=None)):
    mesh = animation.load_frame(frame_i)
    rendered_features = rasterize_vertex_features(cameras, mesh, feature_res, vert_features)
    frames.append(rendered_features.cpu())

import imageio
with imageio.get_writer(Path('outs/deadpool.gif'), mode='I', loop=0) as writer:
    for frame in frames:
        frame_pil = TF.to_pil_image(frame)
        writer.append_data(frame_pil)