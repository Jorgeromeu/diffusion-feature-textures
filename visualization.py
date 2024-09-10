from einops import rearrange
from torch import Tensor
import faiss
import numpy as np

def reduce_feature_map(feature_map: Tensor):

    D, H, W = feature_map.shape

    # reshape to flat
    feature_flat = rearrange(feature_map, 'd h w -> (h w) d')

    # fit PCA matrix
    pca = faiss.PCAMatrix(D, 3)
    pca.train(feature_flat)

    # apply PCA matrix
    reduced_flat = pca.apply(feature_flat)

    for c in range(3):
        minval = reduced_flat[:, c].min()
        maxval = reduced_flat[:, c].max()
        reduced_flat[:, c] = (reduced_flat[:, c] - minval) / (maxval - minval) 

    # reshape to square
    reduced = rearrange(reduced_flat, '(h w) d -> d h w', h=H, w=W)
    reduced = Tensor(reduced)

    return reduced
