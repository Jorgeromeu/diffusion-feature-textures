"""
Description:
This Notebook demonstrates how to extract and visualize the intermediate
activations of StableDiffusion
"""

# %% Imports
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

# %% Load Pipeline
sd_repo = "CompVis/stable-diffusion-v1-4"        
device = 'cuda:0'

pipe = FeaturePipeline.from_pretrained(sd_repo).to(device)


# %% Generate Image
gen = torch.Generator(device=pipe.device)
gen.manual_seed(3)

out = pipe(
    ['Deadpool'],
    num_steps=25,
    generator=gen,
    feature_level=2,
    feature_timestep=0
)
out.images[0]


# %% Extract and visualize
def ordered_sample(lst, N):
    """
    Sample N elements from a list in order.
    """
    step_size = len(lst) // (N - 1)
    # Get the sample by slicing the list
    sample = [lst[i * step_size] for i in range(N - 1)]
    sample.append(lst[-1])  # Add the last element
    return sample

def sd_features_plot(pipe, n_timesteps=5, scale=3):

    levels = pipe.feature_levels
    timesteps = ordered_sample(sorted(list(pipe.feature_timesteps)), n_timesteps)

    n_levels = len(levels)
    n_timesteps = len(timesteps)


    fig, axs = plt.subplots(n_levels, n_timesteps, figsize=(scale*n_timesteps, scale*n_levels))

    pca_matrices = []

    for level in levels:
        level_features = []
        for timestep in timesteps:
            feature = pipe.sd_features[timestep, level]
            level_features.append(feature)
        
        level_features = np.concatenate(level_features)
        level_features = rearrange(level_features, 't d h w -> (t h w) d')

        pca = faiss.PCAMatrix(level_features.shape[-1], 3)
        pca.train(level_features)
        pca_matrices.append(pca) 

    for level_i, level in enumerate(levels):
        for timestep_i, timestep in enumerate(timesteps):

            # get axis
            ax = axs[level_i, timestep_i]
            ax.set_xticks([])
            ax.set_yticks([])

            if level_i == 0:
                ax.set_title(f"t={timestep}")

            if timestep_i == 0:
                ax.set_ylabel(f"level={level}")

            # get feature 
            feature = pipe.sd_features[timestep, level]

            _, d, h, w = feature.shape

            # apply learned PCA
            feature_flat = rearrange(feature[0], 'd h w -> (h w) d')
            rgb_flat = pca_matrices[level_i].apply(feature_flat)

            # scale for visualization 
            scaler = MinMaxScaler()
            rgb_flat_scaled = scaler.fit_transform(rgb_flat)

            # reshape to square
            feature_rgb = rearrange(rgb_flat_scaled, '(h w) d -> h w d', h=h, w=w)
            ax.imshow(feature_rgb) 

    plt.tight_layout()
        

sd_features_plot(pipe, n_timesteps=6, scale=2)
plt.savefig('../outs/sd_features.png', dpi=300)

# %%
