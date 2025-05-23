from typing import List, Tuple

import torch
import torchvision.transforms.functional as TF
from attr import dataclass
from einops import rearrange
from jaxtyping import Float
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.renderer import (
    CamerasBase,
    MeshRasterizer,
    MeshRenderer,
    RasterizationSettings,
    TexturesUV,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.structures import Meshes
from torch import Tensor, nn

from text3d2video.util import sample_feature_map_ndc
from text3d2video.utilities.ipython_utils import display_ims


class TextureShader(nn.Module):
    """
    Simple shader, that returns textured colors of mesh, no shading
    """

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs):
        colors = meshes.sample_textures(fragments)
        mask = fragments.pix_to_face > 0
        output = torch.zeros_like(colors)
        output[mask] = colors[mask]
        output = output[:, :, :, 0, :]
        output = rearrange(output, "b h w c -> b c h w")
        return output


class UVShader(nn.Module):
    """
    Simple shader that returns the rendered UV coordiantes of mesh
    """

    def forward(self, fragments: Fragments, verts_uvs, faces_uvs):
        face_vert_uvs = verts_uvs[faces_uvs]
        pixel_uvs = interpolate_face_attributes(
            fragments.pix_to_face, fragments.bary_coords, face_vert_uvs
        )

        pixel_uvs = rearrange(pixel_uvs, "b h w 1 c -> b c h w")
        return pixel_uvs


def zbuf_to_depth_map(zbuf):
    """
    Convert from zbuf to depth map
    """

    max_depth = zbuf.max()
    indices = zbuf == -1
    zbuf = max_depth - zbuf
    zbuf[indices] = 0
    max_depth = zbuf.max()
    zbuf = zbuf / max_depth
    return zbuf


def make_mesh_rasterizer(
    cameras=None,
    resolution=512,
    faces_per_pixel=1,
    blur_radius=0,
    bin_size=0,
    clip_barycentric_coords=True,
):
    raster_settings = RasterizationSettings(
        image_size=resolution,
        faces_per_pixel=faces_per_pixel,
        blur_radius=blur_radius,
        bin_size=bin_size,
        clip_barycentric_coords=clip_barycentric_coords,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    return rasterizer


def make_mesh_renderer(
    resolution=512,
    faces_per_pixel=1,
    blur_radius=0,
    bin_size=0,
    shader=None,
    cameras=None,
):
    rasterizer = make_mesh_rasterizer(
        resolution=resolution,
        faces_per_pixel=faces_per_pixel,
        blur_radius=blur_radius,
        bin_size=bin_size,
        cameras=cameras,
    )

    if shader is None:
        shader = TextureShader()

    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    return renderer


def render_depth_map(meshes, cameras, resolution=512, chunk_size=30):
    rasterizer = make_mesh_rasterizer(resolution=resolution)
    indices = torch.arange(0, len(meshes))

    all_depth_maps = []
    for chunk_indices in torch.split(indices, chunk_size):
        chunk_meshes = meshes[chunk_indices]
        chunk_cameras = cameras[chunk_indices]
        fragments = rasterizer(chunk_meshes, cameras=chunk_cameras)
        depth_maps = fragments.zbuf
        depth_maps = zbuf_to_depth_map(depth_maps)
        depth_maps = [TF.to_pil_image(depth_map[:, :, 0]) for depth_map in depth_maps]
        all_depth_maps.extend(depth_maps)

    return all_depth_maps


def render_rgb_uv_map(
    meshes, cameras, verts_uvs, faces_uvs, resolution=512, chunk_size=30
):
    rasterizer = make_mesh_rasterizer(resolution=resolution)
    shader = UVShader()

    all_uv_maps = []
    for i in range(0, len(meshes)):
        mesh = meshes[i]
        cam = cameras[i]
        fragments = rasterizer(mesh, cameras=cam)

        # render uv map
        pixel_uvs = shader(fragments, verts_uvs, faces_uvs)
        zero_channel = torch.zeros_like(pixel_uvs[:, 0:1, :, :])
        pixel_uvs_rgb = torch.cat([pixel_uvs, zero_channel], dim=1)

        frame_rgb = TF.to_pil_image(pixel_uvs_rgb[0])
        all_uv_maps.append(frame_rgb)

    return all_uv_maps


def make_repeated_uv_texture(
    uv_map: Float[Tensor, "h w c"],
    faces_uvs: Tensor,
    verts_uvs: Tensor,
    N=1,
    sampling_mode="bilinear",
):
    extended_uv_map = uv_map.to(torch.float32)  # pt3d requires float32 for textures
    extended_uv_map = extended_uv_map.unsqueeze(0).expand(N, -1, -1, -1)
    extended_faces_uvs = faces_uvs.unsqueeze(0).expand(N, -1, -1)
    extended_verts_uvs = verts_uvs.unsqueeze(0).expand(N, -1, -1)
    return TexturesUV(
        extended_uv_map,
        extended_faces_uvs,
        extended_verts_uvs,
        sampling_mode=sampling_mode,
    )


def shade_meshes(
    shader,
    texture: TexturesUV,
    meshes: Meshes,
    fragments: List[Fragments],
):
    renders = []
    for mesh, frags in zip(meshes, fragments):
        mesh.textures = texture
        render = shader(frags, mesh)[0]
        renders.append(render)

    renders = torch.stack(renders)

    return renders


def shade_texture(
    meshes,
    fragments,
    uv_map,
    verts_uvs,
    faces_uvs,
    sampling_mode="bilinear",
    return_pil=False,
):
    renders = []

    shader = TextureShader()
    meshes_copy = meshes.clone()
    texture = make_repeated_uv_texture(
        uv_map, faces_uvs, verts_uvs, sampling_mode=sampling_mode
    )
    for mesh, frags in zip(meshes_copy, fragments):
        mesh.textures = texture
        render = shader(frags, mesh)[0]
        renders.append(render)

    renders = torch.stack(renders)

    if return_pil:
        renders = [TF.to_pil_image(r.cpu()) for r in renders]

    return renders


def render_texture(
    meshes,
    cameras,
    uv_map,
    verts_uvs,
    faces_uvs,
    resolution=512,
    sampling_mode="bilinear",
    return_pil=False,
):
    n_frames = len(cameras)
    texture = make_repeated_uv_texture(
        uv_map, faces_uvs, verts_uvs, N=n_frames, sampling_mode=sampling_mode
    )
    renderer = make_mesh_renderer(cameras=cameras, resolution=resolution)
    meshes_copy = meshes.clone()
    meshes_copy.textures = texture
    renders = renderer(meshes_copy, cameras=cameras)

    if return_pil:
        renders = [TF.to_pil_image(r.cpu()) for r in renders]

    return renders


def compute_autoregressive_update_masks(
    cams,
    meshes,
    projections,
    quality_maps,
    uv_res: int,
    verts_uvs,
    faces_uvs,
    quality_factor=1.5,
):
    """
    Given a sequence of cameras, compute the masks denoting for each render, the parts to update, according to what has already been seen, and image space coordinates
    """

    quality_texture = torch.ones(uv_res, uv_res, 1).cuda() * 10000

    better_quality_masks = []

    for i in range(len(cams)):
        quality_map = quality_maps[i]
        proj = projections[i]
        mesh = meshes[i]
        cam = cams[i]

        # render quality texture
        rendered_quality = render_texture(
            mesh, cam, quality_texture, verts_uvs, faces_uvs
        )[0][0].cpu()

        # mask
        mask = quality_map < rendered_quality / quality_factor
        better_quality_masks.append(mask)

        # sample quality map for view i at prjoected texel locations
        qualities = sample_feature_map_ndc(quality_map.unsqueeze(0).cuda(), proj.xys)

        # get current-best qualities at projected texel locations
        current_qualities = quality_texture[proj.uvs[:, 1], proj.uvs[:, 0]]

        # mask indicating good quality
        better_quality = qualities > current_qualities
        update_mask = better_quality

        update_mask = ~update_mask
        update_mask = update_mask[:, 0]
        update_mask = update_mask

        # update qualities
        update_uvs = proj.uvs[update_mask]
        update_qualities = qualities[update_mask]
        quality_texture[update_uvs[:, 1], update_uvs[:, 0]] = update_qualities

    better_quality_masks = torch.stack(better_quality_masks)

    return better_quality_masks


def downsample_masks(masks: Tensor, size: Tuple[int], thresh=0.8):
    masks = torch.unsqueeze(masks, 1).float()
    masks_resized = TF.resize(masks, size, interpolation=TF.InterpolationMode.BILINEAR)
    masks_resized = masks_resized > thresh
    masks_resized = masks_resized.squeeze(1).cpu().float()
    return masks_resized


def display_frags(
    frags: Fragments,
    show_zbuf=True,
    show_pix_to_face=True,
    show_bary_coords=True,
    show_dists=True,
    show=True,
):
    ims = []
    titles = []

    if show_zbuf:
        ims.append(frags.zbuf[0, ..., 0].cpu())
        titles.append("zbuf")

    if show_pix_to_face:
        ims.append(frags.pix_to_face[0, ..., 0].cpu())
        titles.append("pix_to_face")

    if show_bary_coords:
        ims.append(frags.bary_coords[0, :, :, 0, :].cpu())
        titles.append("bary_coords")

    if show_dists:
        dists = frags.dists[0, ..., 0].cpu()
        ims.append(dists)
        titles.append("dists")

    return display_ims(ims, titles=titles, show=show)


@dataclass
class AnimSequence:
    cams: CamerasBase
    meshes: Meshes
    verts_uvs: Tensor
    faces_uvs: Tensor

    def __len__(self):
        return len(self.cams)

    def render_depth_maps(self):
        """
        Returns a list of depth maps for each frame in the animation sequence.
        """
        return render_depth_map(self.meshes, self.cams)

    def render_rgb_uv_maps(self):
        return render_rgb_uv_map(self.meshes, self.cams, self.verts_uvs, self.faces_uvs)

    def render_texture(
        self,
        texture: Tensor,
        resolution=512,
        sampling_mode="bilinear",
        return_pil=False,
    ):
        return render_texture(
            self.meshes,
            self.cams,
            texture,
            self.verts_uvs,
            self.faces_uvs,
            resolution=resolution,
            sampling_mode=sampling_mode,
            return_pil=return_pil,
        )


def resize_frags(frags: Fragments):
    assert frags.pix_to_face.shape[-1] == 1, "only supports K=1"


def deconstruct_frags(frags: Fragments):
    assert frags.pix_to_face.shape[-1] == 1, "only supports K=1"
    assert frags.pix_to_face.shape[0] == 1, "only supports B=1"

    pix2face = frags.pix_to_face[0]
    bary_cords = frags.bary_coords[0, :, :, 0, :]
    zbuf = frags.zbuf[0]
    dists = frags.dists[0]
    return pix2face, bary_cords, zbuf, dists


def reconstruct_frags(pix2face, bary_cords, zbuf, dists):
    assert pix2face.shape[0] == 1, "only supports B=1"
    assert pix2face.shape[-1] == 1, "only supports K=1"

    pix2face = pix2face.unsqueeze(0)
    bary_cords = bary_cords.unsqueeze(0).unsqueeze(-2)
    zbuf = zbuf.unsqueeze(0)
    dists = dists.unsqueeze(0)
    return Fragments(
        pix_to_face=pix2face,
        bary_coords=bary_cords,
        zbuf=zbuf,
        dists=dists,
    )


def downsample_frags(frags: Fragments, factor: int):
    pix2face, bary_coords, zbuf, dists = deconstruct_frags(frags)

    # use zbuf to get the index of which pixel to keep
    zbuf_no_bg = zbuf.clone()
    zbuf_no_bg[zbuf == -1] = 1000

    # Blockify the tensors into chunks of factor x factor
    def blockify(tensor, factor):
        h, w, c = tensor.shape
        assert h % factor == 0 and w % factor == 0

        return rearrange(
            tensor, "(h f1) (w f2) c -> h w (f1 f2) c", f1=factor, f2=factor
        )

    pix2face_blocks = blockify(pix2face, factor)
    zbuf_blocks = blockify(zbuf, factor)
    bary_blocks = blockify(bary_coords, factor)
    dists_blocks = blockify(dists, factor)
    zbuf_no_bg_blocks = blockify(zbuf_no_bg, factor)

    _, min_idx = zbuf_no_bg_blocks.min(dim=-2)

    def gather_by_index(blocks, idx):
        H, W, B, C = blocks.shape
        idx = idx.expand(-1, -1, C).unsqueeze(2)  # (H, W, 1, C)
        return torch.gather(blocks, 2, idx).squeeze(2)  # (H, W, C)

    pix2face_low = gather_by_index(pix2face_blocks, min_idx)
    bary_low = gather_by_index(bary_blocks, min_idx)
    dists_low = gather_by_index(dists_blocks, min_idx)
    zbuf_low = gather_by_index(zbuf_blocks, min_idx)

    pix2face_low_final = pix2face_low.unsqueeze(0)
    bary_low_final = bary_low.unsqueeze(0).unsqueeze(-2)
    zbuf_low_final = zbuf_low.unsqueeze(0)
    dists_low_final = dists_low.unsqueeze(0)
    return Fragments(
        pix_to_face=pix2face_low_final,
        bary_coords=bary_low_final,
        zbuf=zbuf_low_final,
        dists=dists_low_final,
    )
