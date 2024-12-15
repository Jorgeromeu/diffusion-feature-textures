from pathlib import Path

import rerun as rr
import rerun.blueprint as rrb
import torch
from pxr import Usd, UsdGeom
from pytorch3d.renderer import FoVPerspectiveCameras, join_cameras_as_batch
from pytorch3d.structures import Meshes, join_meshes_as_batch
from torch import Tensor

import text3d2video.rerun_util as ru
from text3d2video.camera_utils import focal_length_to_fov
from text3d2video.coord_utils import (
    apply_transform_homogeneous,
    assemble_transform_srt,
    decompose_transform_srt,
)
from text3d2video.rendering import render_depth_map
from text3d2video.usd_utils import triangulate_usd_mesh, usd_uvs_to_pt3d_uvs

# Blender to Pt3d conventions
P_blender_world_to_pt3d_world = Tensor([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
P_blender_cam_to_pt3d_cam = Tensor([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

BLENDER_WORLD_CONVENTION = rr.ViewCoordinates.RFU
BLENDER_CAM_CONVENTION = rr.ViewCoordinates.RUB

if __name__ == "__main__":
    usd_file = Path("/home/jorge/untitled.usdc")

    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(name="3D"),
            rrb.Spatial3DView(name="camera"),
        ),
        collapse_panels=True,
    )

    # init rerun
    rr.init("view_animation", spawn=True)
    rr.send_blueprint(blueprint)
    seq = ru.TimeSequence("frames")
    rr.log("/", BLENDER_WORLD_CONVENTION, static=True)
    render_resolution = 100

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
    triangle_indices = triangulate_usd_mesh(face_vertex_counts, face_vert_indices)

    # read uvs
    uv = Tensor(mesh.GetPrim().GetProperty("primvars:st").Get())
    verts_uvs, faces_uvs = usd_uvs_to_pt3d_uvs(uv, len(triangle_indices))

    # iterate over frames
    start_time = int(stage.GetStartTimeCode())
    end_time = int(stage.GetEndTimeCode())
    frame_indices = list(range(start_time, end_time + 1))

    cams_pt3d = []
    meshes_pt3d = []

    for frame in frame_indices:
        xform_cache = UsdGeom.XformCache(time=frame)

        # camera extrinsics
        c2w = xform_cache.GetLocalToWorldTransform(camera.GetPrim())
        c2w = Tensor(c2w).T
        cam_t, scale, cam_r = decompose_transform_srt(c2w)

        # get camera intrinsics
        focal_length = camera.GetFocalLengthAttr().Get()
        height = camera.GetVerticalApertureAttr().Get()
        width = camera.GetHorizontalApertureAttr().Get()

        scaling_factor = render_resolution / height
        height_px = render_resolution
        width_px = render_resolution
        focal_length_px = focal_length * scaling_factor

        # log camera
        rr.log(
            "camera",
            rr.Pinhole(
                height=height_px,
                width=width_px,
                focal_length=focal_length_px,
                camera_xyz=BLENDER_CAM_CONVENTION,
            ),
        )
        rr.log(
            "camera",
            rr.Transform3D(translation=cam_t, mat3x3=cam_r),
        )

        # mesh to world transform
        m2w = xform_cache.GetLocalToWorldTransform(mesh.GetPrim())
        m2w = Tensor(m2w).T

        # get mesh world coords at frame
        verts = Tensor(mesh.GetPointsAttr().Get(frame)).float()
        verts_world = apply_transform_homogeneous(verts, m2w)

        # log mesh
        mesh_pt3d = Meshes(verts=[verts_world], faces=[triangle_indices])
        rr.log("mesh", ru.pt3d_mesh(mesh_pt3d))

        # mesh world coordinates in pt3d space
        verts_world_pt3d = verts_world @ P_blender_world_to_pt3d_world.T
        mesh_pt3d = Meshes(verts=[verts_world_pt3d], faces=[triangle_indices])

        # c2w in pt3d space
        cam_r_pt3d = P_blender_world_to_pt3d_world @ cam_r @ P_blender_cam_to_pt3d_cam
        cam_t_pt3d = P_blender_world_to_pt3d_world @ cam_t
        c2w_pt3d = assemble_transform_srt(cam_t_pt3d, torch.ones(3), cam_r_pt3d)

        # construct pt3d camera
        w2c_pt3d = c2w_pt3d.inverse()
        t_w2c, _, r_w2c = decompose_transform_srt(w2c_pt3d)

        fov = focal_length_to_fov(focal_length, height)
        cam_pt3d = FoVPerspectiveCameras(
            R=r_w2c.T.unsqueeze(0), T=t_w2c.unsqueeze(0), fov=fov
        )

        render = render_depth_map(
            mesh_pt3d.cuda(), cam_pt3d.cuda(), resolution=render_resolution
        )[0]
        rr.log("camera/render", rr.Image(render))

        cams_pt3d.append(cam_pt3d)
        meshes_pt3d.append(mesh_pt3d)

        seq.step()

    cams_pt3d = join_cameras_as_batch(cams_pt3d)
    meshes_pt3d = join_meshes_as_batch(meshes_pt3d)
