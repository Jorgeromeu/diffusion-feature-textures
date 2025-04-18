from typing import List

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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


def add_zoom_inset(ax: Axes, box: Bbox, width="40%", loc="lower right", color="red"):
    # make inset
    axins = inset_axes(ax, width=width, height=width, loc=loc)
    axins.set_xticks([])
    axins.set_yticks([])

    # Recover image data, and add to new axes
    img = ax.images[0]
    data = img.get_array()
    extent = img.get_extent()
    cmap = img.get_cmap()
    norm = img.norm
    axins.imshow(
        data,
        extent=extent,
        cmap=cmap,
        norm=norm,
    )

    # zoom
    axins.set_xlim(box.x0, box.x1)
    axins.set_ylim(box.y1, box.y0)

    for spine in axins.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(1)

    rect = Rectangle(
        (box.x0, box.y0),
        box.width,
        box.height,
        edgecolor=color,
        facecolor="none",
        linewidth=1,
    )
    ax.add_patch(rect)


def add_inset(ax: Axes, width="40%", loc="lower right"):
    axins = inset_axes(ax, width=width, height=width, loc=loc)
    axins.set_xticks([])
    axins.set_yticks([])
    return axins


def bbox_around_point(point: np.ndarray, width: int = 10) -> Bbox:
    """
    Create a bounding box around a point
    :param point: (2,) tensor of image coordinates [0,1] top-left
    :param size: size of the bounding box
    """
    x0 = point[0] - width / 2
    y0 = point[1] - width / 2
    x1 = point[0] + width / 2
    y1 = point[1] + width / 2

    return Bbox.from_extents(x0, y0, x1, y1)
