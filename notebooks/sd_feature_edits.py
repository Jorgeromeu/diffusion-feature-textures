# %% Imports
from IPython import get_ipython

ipy = get_ipython()
ipy.extension_manager.load_extension('autoreload')
ipy.run_line_magic('autoreload', '2')

import sys
sys.path.append('../')

from feature_pipeline import FeaturePipeline
from sd_feature_extraction import FeatureExtractor
from sklearn.preprocessing import MinMaxScaler
import faiss
import numpy as np
from einops import rearrange
from PIL import Image
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from visualization import reduce_feature_map
from diffusers import StableDiffusionPipeline

# %% Load Pipeline
device = 'cuda:0'
sd_repo = "runwayml/stable-diffusion-v1-5"

pipe = FeaturePipeline.from_pretrained(sd_repo).to(device)

extractor = FeatureExtractor()

module = pipe.unet.up_blocks[0]
extractor.add_save_named_feature_hook("up_block_0", module)

# %%
out = pipe(
    ["Deadpool"],
    generator=torch.Generator(device).manual_seed(0)
)
out.images[0]

# %%
feature = extractor._saved_features["up_block_0"][0][0]
feature.shape