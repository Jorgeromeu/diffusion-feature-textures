import torch
from pytorch3d.renderer import CamerasBase
from pytorch3d.structures import Meshes

from text3d2video.rendering import UVShader, make_mesh_rasterizer
from text3d2video.util import sample_feature_map_ndc


def evaluate_mesh_at_uv(mesh, uv: torch.Tensor, verts_uvs, faces_uvs):
    """
    Args:
        mesh: a PyTorch3D Meshes object (batch size = 1)
        uv: (2,) tensor, the uv coordinate to evaluate (u, v) in [0, 1]

    Returns:
        3D point (3,) tensor
        normal (3,) tensor
    """
    verts = mesh.verts_padded()  # (1, V, 3)
    faces = mesh.faces_padded()  # (1, F, 3)

    verts = verts[0]  # (V, 3)
    faces = faces[0]  # (F, 3)

    # 2. Build UV triangles
    uv_triangles = verts_uvs[faces_uvs]  # (F, 3, 2)

    # 3. For each triangle, check if UV is inside
    # Compute barycentric coordinates
    def barycentric_coords(p, tri):
        v0 = tri[1] - tri[0]
        v1 = tri[2] - tri[0]
        v2 = p - tri[0]
        d00 = (v0 * v0).sum()
        d01 = (v0 * v1).sum()
        d11 = (v1 * v1).sum()
        d20 = (v2 * v0).sum()
        d21 = (v2 * v1).sum()
        denom = d00 * d11 - d01 * d01
        v = (d11 * d20 - d01 * d21) / denom
        w = (d00 * d21 - d01 * d20) / denom
        u = 1.0 - v - w
        return torch.stack([u, v, w], dim=-1)

    # Vectorize
    uv = uv.unsqueeze(0).unsqueeze(0)  # (1, 1, 2)
    tri0 = uv_triangles[:, 0, :]  # (F, 2)
    tri1 = uv_triangles[:, 1, :]
    tri2 = uv_triangles[:, 2, :]
    v0 = tri1 - tri0  # (F, 2)
    v1 = tri2 - tri0  # (F, 2)
    v2 = uv[:, :, :] - tri0.unsqueeze(0)  # (1, F, 2)

    d00 = (v0 * v0).sum(dim=-1)  # (F,)
    d01 = (v0 * v1).sum(dim=-1)
    d11 = (v1 * v1).sum(dim=-1)
    d20 = (v2 * v0.unsqueeze(0)).sum(dim=-1)[0]
    d21 = (v2 * v1.unsqueeze(0)).sum(dim=-1)[0]
    denom = d00 * d11 - d01 * d01

    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1.0 - v - w
    bary_coords = torch.stack([u, v, w], dim=-1)  # (F, 3)

    # 4. Find which triangle contains the UV
    mask = (bary_coords >= 0.0).all(dim=-1)
    if not mask.any():
        raise ValueError("UV coordinate does not lie inside any face.")

    face_idx = torch.where(mask)[0][0]  # Take the first face that matches

    # 5. Get the corresponding 3D triangle
    tri_verts_idx = faces[face_idx]  # (3,)
    tri_verts = verts[tri_verts_idx]  # (3, 3)

    # 6. Interpolate 3D position
    bary = bary_coords[face_idx]  # (3,)
    point_3d = (tri_verts * bary.unsqueeze(-1)).sum(dim=0)

    return point_3d


def uv_at_ndc_coords(
    cam: CamerasBase, mesh: Meshes, verts_uvs, faces_uvs, ndc_coords: torch.Tensor
):
    rasterizer = make_mesh_rasterizer(resolution=100)
    shader = UVShader()

    fragments = rasterizer(mesh, cameras=cam)
    pixel_uvs = shader(fragments, verts_uvs, faces_uvs)
    uv_coords = sample_feature_map_ndc(pixel_uvs[0], ndc_coords)
    return uv_coords
