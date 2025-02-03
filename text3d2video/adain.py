from math import sqrt

from cv2 import normalize
from einops import rearrange
from jaxtyping import Float
from torch import Tensor


def calc_channel_mean_std_2D(feat: Float[Tensor, "B C H W"], eps=1e-5):
    """
    Calculate mean and std for each channel
    :param feat: feature tensor with shape (B, C, H, W)
    :param eps: a small value added to the variance to avoid divide-by-zero.
    :return: mean and std (B, C, 1, 1)
    """

    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert len(size) == 4
    N, C, _, _ = size
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain_2D(
    content_feat: Float[Tensor, "B C H W"], style_feat: Float[Tensor, "B C H W"]
) -> Tensor:
    assert content_feat.size()[:2] == style_feat.size()[:2]
    size = content_feat.size()

    # compute mean and std for each channel in content and style
    style_mean, style_std = calc_channel_mean_std_2D(style_feat)
    content_mean, content_std = calc_channel_mean_std_2D(content_feat)

    # normalize content feature to have mean 0 and std 1
    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(
        size
    )

    # re-scale the normalized feature with the std and mean of style feature
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def calc_channel_mean_std_1D(feat: Float[Tensor, "B T C"], eps=1e-5):
    mean = feat.mean(dim=1, keepdim=False)
    std = feat.std(dim=1) + 1e-6
    return mean.unsqueeze(1), std.unsqueeze(1)


def adain_1D(content_feat: Float[Tensor, "B T C"], style_feat: Float[Tensor, "B T C"]):
    style_mean, style_std = calc_channel_mean_std_1D(style_feat)
    content_mean, content_std = calc_channel_mean_std_1D(content_feat)

    normalized_feat = (content_feat - content_mean) / content_std
    return normalized_feat * style_std + style_mean
