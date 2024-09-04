# %% Imports
from IPython import get_ipython
from einops import rearrange


ipy = get_ipython()
ipy.extension_manager.load_extension('autoreload')
ipy.run_line_magic('autoreload', '2')

import sys
sys.path.append('../')

from feature_pipeline import FeaturePipeline
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from visualization import reduce_feature_map

# %%
sd_repo = "CompVis/stable-diffusion-v1-4"        
device = 'cuda:0'

pipe = FeaturePipeline.from_pretrained(sd_repo).to(device)
# %%
gen = torch.Generator(device=pipe.device)
gen.manual_seed(0)

out = pipe(
    ['Deadpool'],
    num_steps=25,
    generator=gen,
    feature_level=2,
    feature_timestep=0
)
out.images[0]

feature = pipe.out_feature[0]
feature_rgb = reduce_feature_map(feature)

plt.imshow(feature_rgb.permute(1,2,0))
