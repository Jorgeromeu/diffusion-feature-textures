from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb
import torch
from pxr import Usd, UsdGeom
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Meshes
from torch import Tensor

import text3d2video.rerun_util as ru
from text3d2video.rendering import render_depth_map
from text3d2video.util import ordered_sample


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


# Blender to Pt3d conventions
P_blender_world_to_pt3d_world = Tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
P_blender_cam_to_pt3d_cam = Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

BLENDER_world_convention = rr.ViewCoordinates.RFU
BLENDER_cam_convention = rr.ViewCoordinates.RUB

if __name__ == "__main__":
    usd_file = Path("/home/jorge/untitled.usdc")

    blender_xyz = rr.ViewCoordinates.RUB
    pt3d_xyz = rr.ViewCoordinates.LUF

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D"),
            rrb.Spatial2DView(
                name="2D", contents=["/camera/render"], origin=["camera"]
            ),
        ),
        collapse_panels=True,
    )

    # init rerun
    rr.init("view_animation", spawn=True)
    rr.send_blueprint(blueprint)
    seq = ru.TimeSequence("frames")
    rr.log("/", BLENDER_world_convention, static=True)

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

    frames = list(range(start_time, end_time + 1))

    for frame in ordered_sample(frames, 100):
        xform_cache = UsdGeom.XformCache(time=frame)

        # get camera extrinsics
        c2w = xform_cache.GetLocalToWorldTransform(camera.GetPrim())
        c2w = Tensor(c2w).T

        # get camera intrinsics
        focal_length = camera.GetFocalLengthAttr().Get()
        height = camera.GetVerticalApertureAttr().Get()
        width = camera.GetHorizontalApertureAttr().Get()

        resolution = 100

        scaling_factor = resolution / height
        cam_height = resolution
        cam_width = resolution
        focal_length = focal_length * scaling_factor
        # log camera
        cam_t, scale, cam_r = decompose_transform_srt(c2w)
        rr.log(
            "camera",
            rr.Pinhole(
                height=cam_height,
                width=cam_width,
                focal_length=focal_length,
                camera_xyz=blender_xyz,
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
        verts_world = apply_transform_homogeneous(verts, m2w)

        # log mesh
        mesh_pt3d = Meshes(verts=[verts_world], faces=[faces])
        verts = mesh_pt3d.verts_list()[0].cpu()
        faces = mesh_pt3d.faces_list()[0].cpu()
        normals = mesh_pt3d.verts_normals_list()[0].cpu()
        rr.log(
            "mesh",
            rr.Mesh3D(
                vertex_positions=verts, triangle_indices=faces, vertex_normals=normals
            ),
        )

        # world coordinates in pt3d
        verts_world_pt3d = verts_world @ P_blender_world_to_pt3d_world.T
        mesh_pt3d = Meshes(verts=[verts_world_pt3d], faces=[faces])

        cam_r_pt3d = P_blender_world_to_pt3d_world @ cam_r @ P_blender_cam_to_pt3d_cam
        cam_t_pt3d = P_blender_world_to_pt3d_world @ cam_t

        c2w_pt3d = assemble_transform_srt(cam_t_pt3d, torch.ones(3), cam_r_pt3d)

        w2c_pt3d = c2w_pt3d.inverse()
        t_w2c, _, r_w2c = decompose_transform_srt(w2c_pt3d)

        cam_pt3d = FoVPerspectiveCameras(
            R=r_w2c.T.unsqueeze(0), T=t_w2c.unsqueeze(0), fov=40
        )

        render = render_depth_map(
            mesh_pt3d.cuda(), cam_pt3d.cuda(), resolution=resolution
        )[0]
        rr.log("camera/render", rr.Image(render))

        seq.step()
