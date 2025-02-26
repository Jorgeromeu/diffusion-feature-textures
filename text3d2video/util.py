from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from einops import rearrange
from jaxtyping import Float
from pytorch3d.io import load_obj
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    CamerasBase,
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings,
)
from pytorch3d.structures import Meshes
from torch import Tensor

from text3d2video.rendering import make_feature_renderer
from text3d2video.utilities.camera_placement import turntable_extrinsics
from text3d2video.utilities.mesh_processing import normalize_meshes


def read_obj_uvs(obj_path: str, device="cuda"):
    _, faces, aux = load_obj(obj_path)
    verts_uvs = aux.verts_uvs.to(device)
    faces_uvs = faces.textures_idx.to(device)
    return verts_uvs, faces_uvs


def ordered_sample_indices(lst, n):
    """
    Sample n elements from a list in order, return indices
    """

    if n == 0:
        return []
    if n == 1:
        return [0]
    if n >= len(lst):
        return list(range(len(lst)))
    # Calculate the step size based on list length and number of samples
    step = (len(lst) - 1) / (n - 1)
    # Use the calculated step to select indices
    indices = [round(i * step) for i in range(n)]
    return indices


def ordered_sample(lst, n):
    """
    Sample n elements from a list in order.
    """
    indices = ordered_sample_indices(lst, n)
    return [lst[i] for i in indices]


def project_vertices_to_features(
    mesh: Meshes, cam: CamerasBase, feature_map: Tensor, mode="nearest"
):
    feature_dim, _, _ = feature_map.shape

    # rasterize mesh
    raster_settings = RasterizationSettings(
        image_size=512,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(cameras=cam, raster_settings=raster_settings)
    fragments = rasterizer(mesh)

    # get visible faces
    visible_faces = rearrange(fragments.pix_to_face[0], "h w 1 -> (h w)").unique()

    # visible verts, are the vertices of the visible faces
    visible_vert_indices = mesh.faces_list()[0][visible_faces].flatten()
    verts = mesh.verts_list()[0]
    visible_verts = verts[visible_vert_indices]

    # TODO extract projected points from rasterization pass output, no need to do it again
    # project points to NDC
    visible_points_ndc = cam.transform_points_ndc(visible_verts).cpu()

    # extract features for each projected vertex
    visible_point_features = sample_feature_map_ndc(
        feature_map, visible_points_ndc[:, 0:2], mode
    ).to(verts)

    # construct vertex features tensor
    vert_features = torch.zeros(mesh.num_verts_per_mesh()[0], feature_dim).to(verts)
    vert_features[visible_vert_indices] = visible_point_features

    return vert_features


def project_visible_verts_to_cameras(meshes: Meshes, cameras: CamerasBase):
    # rasterize mesh, to get visible verts
    raster_settings = RasterizationSettings(
        image_size=600,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    fragments = rasterizer(meshes)

    vert_xys = []
    vert_indices = []
    packed_vert_cnt = 0

    for view in range(len(meshes)):
        pix_to_face = fragments.pix_to_face[view]
        mask = pix_to_face > 0
        visible_face_indices = pix_to_face[mask]
        visible_face_indices = visible_face_indices.unique()
        visible_faces = meshes.faces_packed()[visible_face_indices]

        visible_vert_indices = torch.unique(visible_faces)
        visible_verts = meshes.verts_packed()[visible_vert_indices]

        visible_verts_ndc = cameras[view].transform_points_ndc(visible_verts)
        visible_verts_xy = visible_verts_ndc[:, 0:2]
        visible_verts_xy[:, 0] *= -1

        vert_indices.append(visible_vert_indices - packed_vert_cnt)
        vert_xys.append(visible_verts_xy)

        packed_vert_cnt += meshes.num_verts_per_mesh()[view]

    return vert_xys, vert_indices


def aggregate_features_precomputed_vertex_positions(
    feature_maps: Float[Tensor, "n c h w"],
    n_verts: int,
    vertex_positions: List[Float[Tensor, "v 3"]],
    vertex_indices: List[Float[Tensor, "v"]],  # noqa: F821
    mode="nearest",
    aggregation_type="first",
):
    # initialize empty vertex features
    feature_dim = feature_maps.shape[1]
    vert_features = torch.zeros(n_verts, feature_dim).cuda()
    vert_features_cnt = torch.zeros(n_verts).cuda()

    for frame, feature_map in enumerate(feature_maps):
        # get features for each vertex, for given view
        frame_vert_xys = vertex_positions[frame]
        frame_vert_indices = vertex_indices[frame]
        frame_vert_features = sample_feature_map_ndc(
            feature_map,
            frame_vert_xys,
            mode=mode,
        ).to(vert_features)

        if aggregation_type == "mean":
            vert_features[frame_vert_indices] += frame_vert_features

        elif aggregation_type == "first":
            # update empty entries in vert_features
            mask = torch.all(vert_features[frame_vert_indices] == 0, dim=1)
            vert_features[frame_vert_indices[mask]] = frame_vert_features[mask]

        # count number of features per vertex
        frame_nonzero_features = torch.all(frame_vert_features != 0, dim=1)
        frame_nonzero_indices = frame_vert_indices[frame_nonzero_features]
        vert_features_cnt[frame_nonzero_indices] += 1

    if aggregation_type == "mean":
        vert_features /= torch.clamp(vert_features_cnt, min=1).unsqueeze(1)

    # average features
    return vert_features


def sample_feature_map_ndc(feature_map: Tensor, coords: Tensor, mode="nearest"):
    """
    Sample the feature map at the given coordinates
    :param feature_map: (C, H, W) feature map
    :param coords: (N, 2) coordinates in the range [-1, 1] (NDC)
    :param mode: interpolation mode
    :return: (N, C) sampled features
    """
    coords = coords.clone()
    coords[:, 1] *= -1
    batched_feature_map = rearrange(feature_map, "c h w -> 1 c h w").to(torch.float32)
    grid = rearrange(coords, "n d -> 1 1 n d")
    out = F.grid_sample(batched_feature_map, grid, align_corners=True, mode=mode)
    out_features = rearrange(out, "1 f 1 n -> n f")
    return out_features.to(feature_map)


def blend_features(
    features_original: Tensor,
    features_rendered: Tensor,
    alpha: float,
    channel_dim=0,
):
    # compute mask, where features_rendered is not zero
    masks = torch.sum(features_rendered, dim=channel_dim, keepdim=True) != 0

    # blend features
    blended = alpha * features_rendered + (1 - alpha) * features_original

    original_background = features_original * ~masks
    blended_masked = blended * masks

    # return blended features, where features_rendered is not zero
    return original_background + blended_masked


def assert_valid_tensor_shape(shape: Tuple):
    for expected_len in shape:
        assert isinstance(
            expected_len, (int, type(None), str)
        ), f"Dimension length must be int, None or str, received {expected_len}"


def assert_tensor_shape(
    t: Tensor, shape: tuple[int, ...], named_dim_sizes: dict[str, int] = None
):
    error_str = f"Expected tensor of shape {shape}, got {t.shape}"

    assert_valid_tensor_shape(shape)
    assert t.ndim == len(shape), f"{error_str}, wrong number of dimensions"

    if named_dim_sizes is None:
        named_dim_sizes = {}

    for dim_i, expected_len in enumerate(shape):
        true_len = t.shape[dim_i]

        # any len is allowed for None
        if expected_len is None:
            continue

        # assert same length as other dims with same key
        if isinstance(expected_len, str):
            # if symbol length not saved, save it
            if expected_len not in named_dim_sizes:
                named_dim_sizes[expected_len] = true_len
                continue

            expected_named_dim_size = named_dim_sizes[expected_len]
            assert (
                named_dim_sizes[expected_len] == true_len
            ), f"{error_str}, expected {expected_named_dim_size} for dimension {expected_len}, got {true_len}"

    return named_dim_sizes


def assert_tensor_shapes(tensors, named_dim_sizes: Dict[str, int] = None):
    if named_dim_sizes is None:
        named_dim_sizes = {}

    for tensor, shape in tensors:
        named_dim_sizes = assert_tensor_shape(tensor, shape, named_dim_sizes)


def project_visible_texels_to_camera(
    mesh: Meshes,
    camera: CamerasBase,
    verts_uvs: Tensor,
    faces_uvs: Tensor,
    texture_res: int,
    render_resolution=1000,
) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Project visible UV coordinates to camera pixel coordinates
    :param meshes: Meshes object
    :param cameras: CamerasBase object
    :param verts_uvs: (V, 2) UV coordinates
    :param faces_uvs: (F, 3) face indices for UV coordinates
    :param texture_res: UV_map resolution
    :param render_resolution: render resolution

    Return two corresponding sets of points in the camera and UV space

    :return xy_ndc_coords: (N, 2) pixel coordinates
    :return uv_pix_coords_unique: (N, 2) unique UV coordinates
    """

    # rasterize mesh at high resolution
    raster_settings = RasterizationSettings(
        image_size=render_resolution,
        blur_radius=0.0,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    fragments = rasterizer(mesh, cameras=camera)

    # get visible pixels mask
    mask = fragments.zbuf[:, :, :, 0] > 0

    # for each face, for each vert, uv coord
    faces_verts_uvs = verts_uvs[faces_uvs]

    # interpolate uv coordinates, to get a uv at each pixel
    pixel_uvs = interpolate_face_attributes(
        fragments.pix_to_face, fragments.bary_coords, faces_verts_uvs
    )
    pixel_uvs = pixel_uvs[:, :, :, 0, :]

    # 2D coordinate for each pixel, NDC space
    xs = torch.linspace(-1, 1, render_resolution)
    ys = torch.linspace(1, -1, render_resolution)
    ndc_xs, ndc_ys = torch.meshgrid(xs, ys, indexing="xy")
    ndc_coords = torch.stack([ndc_xs, ndc_ys], dim=-1).to(verts_uvs)

    # for each visible pixel get uv coord and xy coord
    uvs = pixel_uvs[mask]
    xys = ndc_coords[mask[0]]

    # convert continuous cartesian uv coords to pixel coords
    size = (texture_res, texture_res)
    uv_pix_coords = uvs.clone()
    uv_pix_coords[:, 0] = uv_pix_coords[:, 0] * size[0]
    uv_pix_coords[:, 1] = (1 - uv_pix_coords[:, 1]) * size[1]
    uv_pix_coords = uv_pix_coords.long()

    # remove duplicates, and get xy coords for each uv pixel
    texel_coords, coord_pix_indices = unique_with_indices(uv_pix_coords, dim=0)
    texel_xy_coords = xys[coord_pix_indices, :]

    return texel_xy_coords, texel_coords


def project_visible_texels_to_cameras_batched(
    meshes: Meshes,
    cameras: CamerasBase,
    verts_uvs: Tensor,
    faces_uvs: Tensor,
    texture_res: int,
    render_resolution=1000,
):
    # TODO make actually batched?

    frame_xy_coords = []
    frame_uv_coords = []
    for mesh, cam in zip(meshes, cameras):
        xy_coords, uv_coords = project_visible_texels_to_camera(
            mesh,
            cam,
            verts_uvs,
            faces_uvs,
            texture_res=texture_res,
            render_resolution=render_resolution,
        )
        torch.cuda.empty_cache()
        frame_xy_coords.append(xy_coords)
        frame_uv_coords.append(uv_coords)

    return frame_xy_coords, frame_uv_coords


def unique_with_indices(tensor: Tensor, dim: int = 0) -> Tuple[Tensor, Tensor]:
    unique, inverse = torch.unique(tensor, sorted=True, return_inverse=True, dim=dim)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    unique_indices = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return unique, unique_indices


def aggregate_feature_maps(
    feature_maps: Float[Tensor, "n c h w"],
    uv_resolution: int,
    vertex_positions: List[Float[Tensor, "v 3"]],
    vertex_indices: List[Float[Tensor, "v"]],  # noqa: F821
    interpolation_mode="nearest",
):
    # initialize empty uv map
    feature_dim = feature_maps.shape[1]
    uv_map = torch.zeros(uv_resolution, uv_resolution, feature_dim).to(feature_maps)

    for i, feature_map in enumerate(feature_maps):
        xy_coords = vertex_positions[i]
        uv_pix_coords = vertex_indices[i]

        # sample features
        colors = sample_feature_map_ndc(
            feature_map,
            xy_coords,
            mode=interpolation_mode,
        ).to(uv_map)

        # update uv map
        uv_map[uv_pix_coords[:, 1], uv_pix_coords[:, 0]] = colors

    # average features
    return uv_map


def render_multiview(mesh: Meshes, n_frames: int = 30, resolution: int = 512):
    mesh_render = normalize_meshes(mesh)
    mesh_render.textures = mesh.textures.clone()
    R, T = turntable_extrinsics(1, torch.linspace(0, 360, n_frames))
    cams = FoVPerspectiveCameras(R=R, T=T, device="cuda", fov=60)
    frame_meshes = mesh_render.extend(len(cams))

    renderer = make_feature_renderer(cameras=cams, resolution=resolution, device="cuda")
    frames = renderer(frame_meshes)

    frames = [TF.to_pil_image(frame.cpu().permute(2, 0, 1)) for frame in frames]
    frames = [
        TF.resize(frame, (512, 512), TF.InterpolationMode.NEAREST) for frame in frames
    ]
    return frames
