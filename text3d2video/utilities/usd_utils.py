import torch
from einops import rearrange
from torch import Tensor


def triangulate_usd_mesh(face_counts: Tensor, face_indices: Tensor):
    idx = 0
    tri_face_indices = []
    for face_count in face_counts:
        indices = face_indices[idx : idx + face_count]
        idx += face_count

        # if face triangular do nothing
        if len(indices) == 3:
            tri_face_indices.append(indices)

        # if quad face, split into two triangles
        if len(indices) == 4:
            tri1_indices = Tensor([0, 1, 2]).long()
            tri2_indices = Tensor([2, 3, 0]).long()
            tri1 = indices[tri1_indices]
            tri2 = indices[tri2_indices]
            tri_face_indices.append(tri1)
            tri_face_indices.append(tri2)

    return torch.stack(tri_face_indices).long()


def usd_uvs_to_pt3d_uvs(uv: Tensor, n_tris: int):
    """
    Convert UVs from USD format to Pytorch3D format
    :param uv: (F*3, 2) tensor specifying UV coordinates for each vertex in each face
    """

    def point_hash(p: Tensor, decimals=15):
        """
        Hash a point to a tuple of floats with given number of precision
        """
        return tuple(round(p[i].item(), decimals) for i in range(2))

    def point_unhash(p: tuple):
        """
        Unhash a point from a tuple of floats
        """
        return torch.Tensor(p)

    face_vert_uvs = rearrange(uv, "(f v) d -> f v d", v=3)

    uv_coord_indices = {}
    uv_coord_idx = 0

    # for each triangle
    faces_uvs = []

    for face_idx in range(n_tris):
        face_indices = []

        # for each verted
        for vert_idx in [0, 1, 2]:
            # get uv coord
            uv_coord = face_vert_uvs[face_idx, vert_idx]
            uv_hash = point_hash(uv_coord)

            if uv_hash not in uv_coord_indices:
                uv_coord_indices[uv_hash] = uv_coord_idx
                uv_coord_idx += 1

            # get index
            index = uv_coord_indices[uv_hash]
            face_indices.append(index)

        faces_uvs.append(face_indices)

    faces_uvs = Tensor(faces_uvs).long()

    # convert dict back to verts_uvs_tensor
    verts_uvs = []
    for point, i in uv_coord_indices.items():
        point = point_unhash(point)

        verts_uvs.append((i, point))

    verts_uvs = sorted(verts_uvs, key=lambda x: x[0])
    verts_uvs = [x[1] for x in verts_uvs]
    verts_uvs = torch.stack(verts_uvs)

    return verts_uvs, faces_uvs
