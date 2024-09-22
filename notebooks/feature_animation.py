# %% Imports
import imageio
from text3d2video.visualization import RgbPcaUtil
from text3d2video.obj_io import OBJAnimation
import torchvision.transforms.functional as TF
from text3d2video.rendering import normalize_depth_map, rasterize, rasterize_vertex_features
from pathlib import Path
from pytorch3d.renderer import FoVPerspectiveCameras, look_at_view_transform
import torch
from PIL import Image
from tqdm import tqdm
from IPython import get_ipython


# %%
device = torch.device("cuda:0")

animation_name = 'dancing'
animation_path = Path('data') / animation_name
animation = OBJAnimation(animation_path)
mesh = animation.load_frame(1)

# %%
R, T = look_at_view_transform(dist=2, azim=0, elev=0)
cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=60)

vert_features = torch.load('outs/mixamo-human_vert_features.pt')

pca = RgbPcaUtil(vert_features.shape[1])
pca.fit(vert_features)
vert_features = pca.features_to_rgb(vert_features)
vert_features = torch.Tensor(vert_features).to(device)

# %%
frames = []
feature_res = 100
for frame_i in tqdm(animation.framenums(sample_n=None)):
    mesh = animation.load_frame(frame_i)
    rendered_features = rasterize_vertex_features(
        cameras, mesh, feature_res, vert_features)
    frames.append(rendered_features.cpu())

# %%
out_path = Path('outs') / f'{animation_name}.gif'
with imageio.get_writer(str(out_path), mode='I', loop=0) as writer:
    for frame in frames:
        frame_pil = TF.to_pil_image(frame)
        frame_pil = frame_pil.resize((512, 512), Image.NEAREST)
        writer.append_data(frame_pil)
