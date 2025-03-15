from typing import List

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from torch import Tensor


def make_gridspec_figure(
    n_rows: int,
    n_cols: int,
    width_ratios: List[int],
    height_ratios: List[int],
    scale: int = 2,
):
    """
    Create a figure with a gridspec layout
    """

    height = sum(height_ratios) + 0.2
    width = sum(width_ratios)

    fig = plt.figure(figsize=(width * scale, height * scale))
    gs = fig.add_gridspec(
        n_rows,
        n_cols,
        width_ratios=width_ratios,
        height_ratios=height_ratios,
        hspace=0.0,
        wspace=0.01,
    )

    return fig, gs


def coord_to_pixel(coord, size):
    """
    Convert cartesian coordinate to pixel coordinates
    :param coord: (2,) tensor of normalized coordinates [0,1]
    :param size: (2,) tensor of image size
    """

    coord_pix = coord.clone()
    coord_pix[0] = coord_pix[0] * size[0]
    coord_pix[1] = coord_pix[1] * size[1]
    coord_pix = coord_pix.long()

    return coord_pix


def add_pixel_marker(
    ax: Axes, coord: Tensor, bg_img_size, grid_img_size, color="lime", opacity=0.8
):
    """
    Add a pixel marker to a maptlotlib axes
    :param ax: matplotlib axes
    :param coord: (2,) tensor of image coordinates [0,1] top-left
    :param bg_img_size: (2,) shape of background image
    : param layer_res: resolution of the layer
    """

    # get coordinate in layer_res pixel space
    coord_pix = coord_to_pixel(coord, grid_img_size)

    # convert to image space
    pixel_w = bg_img_size[0] / grid_img_size[0]
    coord_pix = coord_pix * pixel_w

    # create rect
    rect = Rectangle(coord_pix, height=pixel_w, width=pixel_w, color=color)
    return ax.add_patch(rect)
