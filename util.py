import torch

def ndc_grid(resolution=100,
             corner_aligned=False):
    """
    Return a 2xHxH tensor where each pixel has the NDC coordinates of the pixel
    :param resolution:
    :param corner_aligned
    :return:
    """

    u = 1 if corner_aligned else 1 - (1 / resolution)

    xs = torch.linspace(u, -u, resolution)
    ys = torch.linspace(u, -u, resolution)
    x, y = torch.meshgrid(xs, ys, indexing='xy')

    # stack to two-channel image
    xy = torch.stack([x, y])

    return xy
