# %% Imports
ipy = get_ipython()
ipy.extension_manager.load_extension('autoreload')
ipy.run_line_magic('autoreload', '2')

import sys
sys.path.append('../')

import imageio
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
from file_util import load_frame_obj
from pathlib import Path
from rendering import normalize_depth_map, rasterize, rasterize_vertex_features
from util import feature_per_vertex, sample_feature_map
from diffusion import depth2img_pipe, depth2img

to_pil = transforms.ToPILImage()
to_tensor = transforms.ToTensor()

# %%
from file_util import OBJAnimation

device = torch.device("cuda:0")

animation_name = 'backflip'
animation_path = Path(f'../data/{animation_name}')
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
feature_maps = []
for i in range(4):
    # extract feature map
    feature_map = feature_extractor.get_feature(level=i, timestep=-1)
    feature_map = torch.Tensor(feature_map)

    # reduce feature map to RGB
    feature_map_rgb = reduce_feature_map(feature_map)

    # store feature map 
    feature_maps.append(feature_map_rgb)

# also include the original image
feature_maps.append(to_tensor(img_out))

fig, axs = plt.subplots(1, len(feature_maps))
for i, ax in enumerate(axs):
    ax.imshow(feature_maps[i].permute(1,2,0).cpu().numpy())
    ax.axis('off')

# %%
def render_vertex_features_to_frames(animation, cameras, vert_features, resolution, n_frames):
    frames = []
    feature_res = feature_map_rgb.shape[1]
    for frame_i in tqdm(animation.framenums(sample_n=n_frames)):
        mesh = animation.load_frame(frame_i)
        rendered_features = rasterize_vertex_features(cameras, mesh, resolution, vert_features)
        frames.append(rendered_features.cpu())
    
    return frames

def frames_to_gif(frames, gif_path: Path):
    with imageio.get_writer(gif_path, mode='I', loop=0) as writer:
        for frame in frames:
            frame_pil = to_pil(frame)
            writer.append_data(frame_pil)
    pass

n_frames = 10
for i, map in enumerate(feature_maps):
    vert_features = feature_per_vertex(mesh, cameras, map)

    feature_res = map.shape[1]
    frames = render_vertex_features_to_frames(animation, cameras, vert_features, feature_res, n_frames)
    gif_path = Path(f'../outs/feature_animations/{animation_name}/map_{i}.gif')
    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames_to_gif(frames, gif_path)