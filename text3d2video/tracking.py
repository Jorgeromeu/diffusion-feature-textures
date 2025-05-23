from einops import rearrange
from torch import Tensor

from text3d2video.backprojection import rasterize_uv_mesh
from text3d2video.rendering import UVShader, make_mesh_rasterizer
from text3d2video.util import sample_feature_map_ndc


def uv_coord_at_ndc(cam, mesh, verts_uvs, faces_uvs, xy_coord, raster_res=512):
    rasterizer = make_mesh_rasterizer(resolution=raster_res)
    shader = UVShader()
    frags = rasterizer(mesh, cameras=cam)
    uvs = shader(frags, verts_uvs, faces_uvs)[0]
    return sample_feature_map_ndc(uvs, xy_coord.unsqueeze(0))[0]


def world_coords_at_uvs(mesh, verts_uvs, faces_uvs, uv_coords: Tensor, raster_res=512):
    uv_frags = rasterize_uv_mesh(verts_uvs, faces_uvs, uv_res=raster_res)
    pix_to_face = uv_frags.pix_to_face[..., 0]
    bary_coords = rearrange(uv_frags.bary_coords, "1 h w 1 c -> c h w")

    uv_ndc = uv_coords * 2 - 1

    texel_faces = sample_feature_map_ndc(pix_to_face, uv_ndc, mode="nearest")
    texel_bary = sample_feature_map_ndc(bary_coords, uv_ndc, mode="bilinear")

    faces = mesh.faces_list()[0]
    verts = mesh.verts_list()[0]

    tri_inds = faces[texel_faces.cpu()][0]
    v0, v1, v2 = verts[tri_inds[:, 0]], verts[tri_inds[:, 1]], verts[tri_inds[:, 2]]
    coords_3D = (
        texel_bary[:, 0:1] * v0 + texel_bary[:, 1:2] * v1 + texel_bary[:, 2:3] * v2
    )
    return coords_3D


def ndc_coord_at_uv(cam, mesh, verts_uvs, faces_uvs, uv_coord):
    world_coord = world_coords_at_uvs(
        mesh, verts_uvs, faces_uvs, uv_coord.unsqueeze(0)
    )[0]
    point_ndc = cam.transform_points_ndc(world_coord.unsqueeze(0))[0]
    return point_ndc[0:2]
