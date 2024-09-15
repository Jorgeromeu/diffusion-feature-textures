from einops import rearrange
from torch import Tensor
import faiss
import numpy as np


class RgbPcaUtil:

    """
    Utility class to facilitate performing PCA on high-dimensional features
    and visualizing them as RGB
    """

    def __init__(self, feature_dim: int):
        self.pca = faiss.PCAMatrix(feature_dim, 3)
        self.feature_dim = feature_dim

    def fit(self, features: Tensor):
        """
        Fit PCA matrix to features
        :param features: N x D tensor
        """
        self.pca.train(features)

    def apply(self, features: Tensor):
        """
        Apply PCA matrix to features
        :param features: N x D tensor
        :return reduced: N x 3 tensor
        """
        return self.pca.apply(features)

    def normalize_rgb(self, reduced):
        """
        Normalize reduced features for visualization
        :param reduced: N x 3 tensor
        :return normalized: N x 3 tensor
        """

        # normalize each channel
        for c in range(3):
            minval = reduced[:, c].min()
            maxval = reduced[:, c].max()
            reduced[:, c] = (reduced[:, c] - minval) / (maxval - minval)

        return reduced

    def features_to_rgb(self, features: Tensor):
        """
        Convert features to RGB
        :param features: N x D tensor
        :return rgb: N x 3 tensor
        """
        reduced = self.apply(features)
        return self.normalize_rgb(reduced)

    def feature_map_to_rgb(self, feature_map: Tensor):
        """
        Convert feature map to RGB
        :param feature_map: D x H x W tensor
        :return rgb: H x W x 3 tensor
        """
        _, H, W = feature_map.shape

        # reshape to flat
        feature_flat = rearrange(feature_map, 'c h w -> (h w) c')

        # apply conversion
        reduced_flat = self.features_to_rgb(feature_flat)

        # reshape to square
        reduced = rearrange(reduced_flat, '(h w) c -> c h w', h=H, w=W)
        reduced = Tensor(reduced)

        return reduced
