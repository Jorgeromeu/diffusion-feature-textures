from jaxtyping import Float
from torch import Tensor


def calc_mean_std(feat: Float[Tensor, "B C H W"], eps=1e-5):
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

    # compute mean and std for each channel
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(
        size
    )
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)
