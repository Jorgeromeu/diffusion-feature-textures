import torch
from attr import dataclass
from pytorch3d.renderer import CamerasBase
from pytorch3d.structures import Meshes

from text3d2video.rendering import AnimSequence, make_mesh_rasterizer


@dataclass
class FaceUVGrads:
    du_dx: torch.Tensor
    du_dy: torch.Tensor
    dv_dx: torch.Tensor
    dv_dy: torch.Tensor


def compute_face_uv_grads(
    mesh: Meshes, cam: CamerasBase, verts_uvs, faces_uvs, resolution=512, uv_res=100
):
    """
    For a given mesh/cam pair compute du/dx, dv/dx, du/dy, dv/dy gradients for each face
    """

    faces = mesh.faces_list()[0]
    verts = mesh.verts_list()[0]

    face_verts_uv = verts_uvs[faces_uvs]  # for each face, for each vert, UV
    face_verts = verts[faces]  # for each face, for each vert, xyz

    face_verts_screen = cam.transform_points_screen(
        face_verts, image_size=(resolution, resolution)
    )
    face_verts_xy = (
        face_verts_screen[:, :, :2] - 0.5
    )  # for each face, for each vert, xy

    v0, v1, v2 = face_verts_xy[:, 0], face_verts_xy[:, 1], face_verts_xy[:, 2]  # (F, 2)
    uv0, uv1, uv2 = (
        face_verts_uv[:, 0],
        face_verts_uv[:, 1],
        face_verts_uv[:, 2],
    )

    # Construct screen-space edge vectors
    A = torch.stack([v1 - v0, v2 - v0], dim=-1)  # (F, 2, 2)
    A_inv = torch.linalg.inv(A)  # (F, 2, 2)

    # Construct UV deltas
    du = torch.stack([uv1[:, 0] - uv0[:, 0], uv2[:, 0] - uv0[:, 0]], dim=-1)  # (F, 2)
    dv = torch.stack([uv1[:, 1] - uv0[:, 1], uv2[:, 1] - uv0[:, 1]], dim=-1)  # (F, 2)

    # Compute UV gradients in screen space
    dudxy = torch.bmm(du.unsqueeze(1), A_inv).squeeze(1)  # (F, 2) — du/dx, du/dy
    dvdxy = torch.bmm(dv.unsqueeze(1), A_inv).squeeze(1)  # (F, 2) — dv/dx, dv/dy

    du_dx_face, du_dy_face = dudxy[:, 0], dudxy[:, 1]
    dv_dx_face, dv_dy_face = dvdxy[:, 0], dvdxy[:, 1]

    # Normalize gradients by uv_res
    du_dx_face *= uv_res
    du_dy_face *= uv_res
    dv_dx_face *= uv_res
    dv_dy_face *= uv_res

    return FaceUVGrads(
        du_dx=du_dx_face,
        du_dy=du_dy_face,
        dv_dx=dv_dx_face,
        dv_dy=dv_dy_face,
    )


def compute_face_uv_jacobian_magnitudes(
    mesh, cam, verts_uvs, faces_uvs, resolution=512, uv_res=100
):
    """
    For a given mesh/cam pair compute jacobian magnitude of dUV/dXY
    """

    face_uv_grads = compute_face_uv_grads(mesh, cam, verts_uvs, faces_uvs, resolution)
    return torch.abs(
        face_uv_grads.du_dx * face_uv_grads.dv_dy
        - face_uv_grads.du_dy * face_uv_grads.dv_dx
    )


def compute_face_mip_level(mesh, cam, verts_uvs, faces_uvs, resolution=512, uv_res=100):
    face_uv_grads = compute_face_uv_grads(
        mesh, cam, verts_uvs, faces_uvs, resolution, uv_res
    )

    return (
        torch.sqrt(
            face_uv_grads.du_dx**2
            + face_uv_grads.du_dy**2
            + face_uv_grads.dv_dx**2
            + face_uv_grads.dv_dy**2
        )
        / 2.0
    )


def broadcast_face_attribute(pix_to_face, face_values):
    """
    Broadcast face values to pixels according to pix_to_face mapping.
    """
    H, W = pix_to_face.shape
    device = pix_to_face.device

    attr_image = torch.zeros((H, W), dtype=face_values.dtype, device=device)
    valid = pix_to_face >= 0
    face_idx = pix_to_face[valid]

    attr_image[valid] = face_values[face_idx]
    return attr_image


def render_uv_jacobian_magnitude_map(
    mesh: Meshes, cam: CamerasBase, verts_uvs, faces_uvs, resolution=512, uv_res=512
):
    rasterizer = make_mesh_rasterizer(resolution=resolution)
    frags = rasterizer(mesh, cameras=cam)
    pix_to_face = frags.pix_to_face[0, ..., 0]  # (H, W)

    # jacobian magnitude of each face
    face_jacobians = compute_face_uv_jacobian_magnitudes(
        mesh, cam, verts_uvs, faces_uvs, resolution, uv_res
    )

    return broadcast_face_attribute(pix_to_face, face_jacobians)


def render_mip_level_map(
    mesh: Meshes, cam: CamerasBase, verts_uvs, faces_uvs, resolution=512, uv_res=512
):
    rasterizer = make_mesh_rasterizer(resolution=resolution)
    frags = rasterizer(mesh, cameras=cam)
    pix_to_face = frags.pix_to_face[0, ..., 0]  # (H, W)
    face_mip_levels = compute_face_mip_level(
        mesh, cam, verts_uvs, faces_uvs, resolution, uv_res
    )
    return broadcast_face_attribute(pix_to_face, face_mip_levels)


def view_mip_level(cam, mesh, verts_uvs, faces_uvs, resolution=512, quantile=0.1):
    rho_unitless = render_mip_level_map(
        mesh, cam, verts_uvs, faces_uvs, resolution=resolution, uv_res=1
    )
    rho_unitless = rho_unitless[rho_unitless > 0]
    rho_summary = torch.quantile(rho_unitless, quantile)
    return rho_summary.item()


def view_uv_res(cam, mesh, verts_uvs, faces_uvs, resolution, quantile=0.1):
    # view_mip_level(cam, mesh, verts_uvs, faces_uvs)
    rho_summary = view_mip_level(
        cam, mesh, verts_uvs, faces_uvs, quantile=quantile, resolution=resolution
    )
    good_uv_res = 1 / rho_summary
    return int(good_uv_res)


def seq_max_uv_res(seq: AnimSequence, resolution=64, quantile=0.1):
    return max(
        [
            view_uv_res(c, m, seq.verts_uvs, seq.faces_uvs, resolution, quantile)
            for c, m in zip(seq.cams, seq.meshes)
        ]
    )
