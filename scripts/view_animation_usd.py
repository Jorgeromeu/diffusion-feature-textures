from pathlib import Path

import rerun as rr
import torch
from numpy import mat
from pxr import Usd, UsdGeom
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Meshes
from torch import Tensor

import text3d2video.rerun_util as ru
from text3d2video.rendering import render_depth_map


def triangulate_usd_mesh(face_counts: Tensor, face_indices: Tensor):
    idx = 0
    tri_face_indices = []
    for face_count in face_counts:
        indices = face_indices[idx : idx + face_count]
        idx += face_count

        if len(indices) == 3:
            tri_face_indices.append(indices)

        if len(indices) == 4:
            tri1_indices = Tensor([0, 1, 2]).long()
            tri2_indices = Tensor([2, 3, 0]).long()
            tri1 = indices[tri1_indices]
            tri2 = indices[tri2_indices]
            tri_face_indices.append(tri1)
            tri_face_indices.append(tri2)

    return torch.stack(tri_face_indices)


def decompose_transform_srt(transform: Tensor, transposed=False):
    """
    Decompose a 4x4 homogeneous transform matrix representing a 3D transformation into its translation, scale and rotation components. Applied in the order: scale -> rotation -> translation.
    NOTE: only works for non-negative uniform scales
    """

    transform = transform.clone()

    if transposed:
        transform = transform.t()

    translation = transform[0:3, 3]
    mat_3x3 = transform[0:3, 0:3]
    scale_x = torch.norm(mat_3x3[0])
    scale_y = torch.norm(mat_3x3[1])
    scale_z = torch.norm(mat_3x3[2])
    scale = torch.stack([scale_x, scale_y, scale_z])
    rotation = mat_3x3 / scale
    return translation, scale, rotation


def assemble_transform_srt(translation: Tensor, scale: Tensor, rotation: Tensor):
    mat_3x3 = rotation * scale
    transform = torch.eye(4)
    transform[0:3, 3] = translation
    transform[0:3, 0:3] = mat_3x3
    return transform


def apply_transform_homogeneous(vertices: Tensor, transform: Tensor):
    verts_homog = torch.cat([vertices, torch.ones(vertices.shape[0], 1)], dim=1)
    vertices = verts_homog @ transform.t()
    return vertices[:, :3]


def apply_frame_convention_conversion(matrix_or_point):
    conventionTransform = torch.Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]]).inverse()
    return matrix_or_point @ conventionTransform.T


if __name__ == "__main__":
    usd_file = Path("/home/jorge/untitled.usdc")

    # init rerun
    rr.init("view_animation", spawn=True)
    seq = ru.TimeSequence("frames")

    # open usd file
    stage = Usd.Stage.Open(str(usd_file))

    # find mesh/cam
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)

        if prim.IsA(UsdGeom.Camera):
            camera = UsdGeom.Camera(prim)

    # read mesh topology
    face_vert_indices = Tensor(mesh.GetFaceVertexIndicesAttr().Get()).long()
    face_vertex_counts = Tensor(mesh.GetFaceVertexCountsAttr().Get()).long()

    # compute triangualted indices
    faces = triangulate_usd_mesh(face_vertex_counts, face_vert_indices)

    # iterate over frames
    start_time = int(stage.GetStartTimeCode())
    end_time = int(stage.GetEndTimeCode())
    for frame in range(start_time, end_time + 1):
        xform_cache = UsdGeom.XformCache(time=frame)

        # get camera extrinsics
        c2w = xform_cache.GetLocalToWorldTransform(camera.GetPrim())
        c2w = Tensor(c2w).T

        # get camera intrinsics
        focal_length = camera.GetFocalLengthAttr().Get()
        height = camera.GetVerticalApertureAttr().Get()
        width = camera.GetHorizontalApertureAttr().Get()

        resolution = 512

        # log camera
        cam_t, scale, cam_r = decompose_transform_srt(c2w)
        blender_xyz = rr.ViewCoordinates.RUB
        pt3d_xyz = rr.ViewCoordinates.LUF
        rr.log(
            "camera",
            rr.Pinhole(
                height=resolution,
                width=resolution,
                focal_length=resolution * focal_length,
                camera_xyz=pt3d_xyz,
            ),
        )
        rr.log(
            "camera",
            rr.Transform3D(translation=cam_t, mat3x3=cam_r),
        )

        # get mesh world transform
        m2w = xform_cache.GetLocalToWorldTransform(mesh.GetPrim())
        m2w = Tensor(m2w).T

        # get vert positions at frame
        verts = Tensor(mesh.GetPointsAttr().Get(frame)).float()
        verts = apply_transform_homogeneous(verts, m2w)

        mesh_pt3d = Meshes(verts=[verts], faces=[faces])
        rr.log("mesh", ru.pt3d_mesh(mesh_pt3d))

        # get w2c transform
        w2c = c2w.inverse()
        t_w2c, _, r_w2c = decompose_transform_srt(w2c)
        cam_pt3d = FoVPerspectiveCameras(
            R=r_w2c.T.unsqueeze(0), T=t_w2c.unsqueeze(0), fov=60
        )

        depth_map = render_depth_map(mesh_pt3d.cuda(), cam_pt3d.cuda(), resolution=512)[
            0
        ]

        rr.log("camera", rr.Image(depth_map))

        seq.step()
