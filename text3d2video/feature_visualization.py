import faiss
import numpy as np
import torchvision.transforms.functional as TF
from einops import rearrange
from torch import Tensor


def reduce_feature_map(feature_map: Tensor, output_type="pil"):
    pca = RgbPcaUtil.init_from_features(rearrange(feature_map, "c h w -> (h w) c"))

    if output_type == "pil":
        return pca.feature_map_to_rgb_pil(feature_map)

    return pca.feature_map_to_rgb(feature_map)


class RgbPcaUtil:
    """
    Utility class to facilitate performing PCA on high-dimensional features,
    and normalizing/processing for RGB visualization.
    """

    pca: faiss.PCAMatrix
    feature_dim: int
    upper_percentile: int
    lower_percentile: int

    # min and max values for each channel
    channel_min: Tensor
    channel_max: Tensor

    def __init__(self, feature_dim: int, upper_percentile=99, lower_percentile=1):
        self.pca = faiss.PCAMatrix(feature_dim, 3)
        self.upper_percentile = upper_percentile
        self.lower_percentile = lower_percentile
        self.feature_dim = feature_dim

    @classmethod
    def init_from_features(
        cls, features: Tensor, upper_percentile=99, lower_percentile=1
    ):
        pca = cls(features.shape[1], upper_percentile, lower_percentile)
        pca.fit(features)
        return pca

    def fit(self, features: Tensor):
        """
        Fit PCA matrix to features
        :param features: N x D tensor
        """

        # fit pca matrix
        # pylint: disable=no-value-for-parameter
        self.pca.train(features)

        # compute min and max for each channel
        reduced_features = self.apply(features)
        self.channel_min = np.percentile(
            reduced_features, self.lower_percentile, axis=0
        )
        self.channel_max = np.percentile(
            reduced_features, self.upper_percentile, axis=0
        )

        normalized = self.normalize(reduced_features)
        return normalized

    def apply(self, features: Tensor):
        """
        Apply PCA matrix to features
        :param features: N x D tensor
        :return reduced: N x 3 tensor
        """

        # pylint: disable=no-value-for-parameter
        return self.pca.apply(features)

    def normalize(self, reduced):
        """
        Normalize reduced features for visualization
        :param reduced: N x 3 tensor
        :return normalized: N x 3 tensor
        """

        # normalize each channel according to precomputed min/max
        for c in range(3):
            minval = self.channel_min[c]
            maxval = self.channel_max[c]
            reduced[:, c] = (reduced[:, c] - minval) / (maxval - minval)

        # clip remaining values
        reduced = np.clip(reduced, 0, 1)

        return reduced

    def features_to_rgb(self, features: Tensor):
        """
        Convert features to normalized RGB
        :param features: N x D tensor
        :return rgb: N x 3 tensor
        """
        reduced = self.apply(features)
        return self.normalize(reduced)

    def feature_map_to_rgb(self, feature_map: Tensor):
        """
        Convert feature map to normalized RGB
        :param feature_map: D x H x W tensor
        :return rgb: 3 x H x W tensor
        """
        _, H, W = feature_map.shape

        # reshape to flat
        feature_flat = rearrange(feature_map, "c h w -> (h w) c")

        # apply conversion
        reduced_flat = self.features_to_rgb(feature_flat)

        # reshape to square
        reduced = rearrange(reduced_flat, "(h w) c -> c h w", h=H, w=W)
        reduced = Tensor(reduced)

        return reduced

    def feature_map_to_rgb_pil(self, feature_map: Tensor):
        """
        Convert feature map to normalized RGB PIL image
        :param feature_map: D x H x W tensor
        :return rgb: H x W x 3 tensor
        """
        return TF.to_pil_image(self.feature_map_to_rgb(feature_map))
