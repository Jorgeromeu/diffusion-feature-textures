from diffusers import UNet2DConditionModel
from typing import Dict, List, Set, Tuple
import numpy as np
import torch
from diffusers import DiffusionPipeline

class SDFeatureExtractor:

    # store saved features here 
    _saved_features: Dict[int, list[np.array]]
    _handles: List[torch.utils.hooks.RemovableHandle]

    def __init__(self, pipe: DiffusionPipeline):

        self.unet = pipe.unet

        self._saved_features = dict()

        self.create_hooks()

    def create_hooks(self):
        for level, up_block in enumerate(self.unet.up_blocks): 
            self._saved_features[level] = []
            handle = up_block.register_forward_hook(self._save_feature_hook(level))

    def get_feature(self, level=0, timestep=0):
        return self._saved_features[level][timestep]

    def _save_feature_hook(self, level):
        def hook(module, inp, out):
            out_np = out.cpu().numpy()[0]
            self._saved_features[level].append(out_np)
        return hook

