from typing import Dict, List

import matplotlib.tri as mtri
import numpy as np
from matplotlib import gridspec, patches
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from tomlkit import key_value
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
    axins = inset_axes(
        ax,
        width=width,
        height=width,
        loc=loc,
    )
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


def make_zoom_inset(src_ax: Axes, tgt_ax: Axes, box: Bbox, color="red"):
    # Recover image data, and add to new axes
    img = src_ax.images[0]
    data = img.get_array()
    extent = img.get_extent()
    cmap = img.get_cmap()
    norm = img.norm
    tgt_ax.imshow(
        data,
        extent=extent,
        cmap=cmap,
        norm=norm,
    )

    # zoom
    tgt_ax.set_xlim(box.x0, box.x1)
    tgt_ax.set_ylim(box.y1, box.y0)

    for spine in tgt_ax.spines.values():
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
    src_ax.add_patch(rect)


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


def binary_masks_diff(ax: Axes, mask_before: Tensor, mask_after: Tensor):
    diff = mask_after.int() - mask_before.int()
    ax.imshow(diff == 1, cmap="Greens", alpha=0.8)
    ax.imshow(diff == -1, cmap="Reds", alpha=0.8)


def mpl_uv_triangulation(verts_uvs: Tensor, faces_uvs: Tensor):
    return mtri.Triangulation(
        x=verts_uvs[:, 0].cpu(),
        y=verts_uvs[:, 1].cpu(),
        triangles=faces_uvs.cpu(),
    )


def chunk_backwards(lst, chunk_size=2):
    return [lst[max(0, i - chunk_size) : i] for i in range(len(lst), 0, -chunk_size)][
        ::-1
    ]


# GRID


def _make_grid(shapes: List, depth: int, dim_labels, outer_gs, fig, gap=0.1):
    shape = shapes[0]
    n_rows = shape[0]
    n_cols = shape[1]

    x_dim = depth * 2
    y_dim = depth * 2 + 1
    x_labels = dim_labels.get(x_dim, None)
    y_labels = dim_labels.get(y_dim, None)

    # make Axes
    if len(shapes) == 1:
        gs = gridspec.GridSpecFromSubplotSpec(
            n_rows, n_cols, subplot_spec=outer_gs, wspace=gap, hspace=gap
        )

        axs = np.empty((n_rows, n_cols), dtype=object)
        for i in range(n_rows):
            for j in range(n_cols):
                ax = fig.add_subplot(gs[i, j])
                ax.set_xticks([])
                ax.set_yticks([])
                axs[i, j] = ax

        if y_labels is not None:
            for i, ax in enumerate(axs[0, :]):
                ax.set_title(y_labels[i], fontsize=20)

        if x_labels is not None:
            for i, ax in enumerate(axs[:, 0]):
                ax.set_ylabel(x_labels[i], fontsize=20)

        return axs.squeeze()

    # make another gridspec
    else:
        gs = gridspec.GridSpecFromSubplotSpec(
            n_rows, n_cols, subplot_spec=outer_gs, hspace=gap, wspace=gap
        )

        all_axs = []
        for i in range(n_rows):
            row_axs = []
            for j in range(n_cols):
                cell = gs[i, j]
                axs = _make_grid(shapes[1:], depth + 1, dim_labels, cell, fig, gap=gap)

                # compute subgrid bbox
                bboxes = [ax.get_position(fig.transFigure) for ax in axs.flatten()]
                bbox = Bbox.union(bboxes)

                if y_labels is not None:
                    fig.text(
                        bbox.x0 + bbox.width / 2,
                        bbox.y1 + 0.01,
                        y_labels[j],
                        ha="center",
                        va="bottom",
                    )

                if x_labels is not None:
                    fig.text(
                        bbox.x0 - 0.01,  # slightly to the left of the bbox
                        bbox.y0 + bbox.height / 2,  # vertically centered
                        x_labels[i],
                        ha="right",
                        va="center",
                    )

                row_axs.append(axs)
            all_axs.append(np.stack(row_axs))

        all_axs = np.stack(all_axs)

        return all_axs.squeeze()


def make_grid(shape: List[int], dim_labels: Dict[int, List[str]], scale=1, gap=0.1):
    grid_shapes = chunk_backwards(shape)
    if len(grid_shapes[0]) == 1:
        grid_shapes[0] = [1] + grid_shapes[0]
        dim_labels = {k + 1: v for k, v in dim_labels.items()}

    shapes_np = np.array(grid_shapes)
    height = np.prod(shapes_np[:, 0])
    width = np.prod(shapes_np[:, 1])
    fig = plt.figure(figsize=(width * scale, height * scale), constrained_layout=False)

    gs = gridspec.GridSpec(1, 1, figure=fig)

    return fig, _make_grid(grid_shapes, 0, dim_labels, gs[0], fig, gap=gap)


# Rects


def rectangle_from_bbox(bbox, edgecolor="red", facecolor="none", linewidth=2, **kwargs):
    """Create a Rectangle patch from a Bbox."""
    x0, y0 = bbox.x0, bbox.y0
    width = bbox.width
    height = bbox.height
    return patches.Rectangle(
        (x0, y0),
        width,
        height,
        edgecolor=edgecolor,
        facecolor=facecolor,
        linewidth=linewidth,
        **kwargs,
    )
