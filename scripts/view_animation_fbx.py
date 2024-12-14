from pathlib import Path

import rerun as rr
from pxr import Usd, UsdGeom, UsdSkel
from pytorch3d.renderer import FoVPerspectiveCameras
from torch import Tensor

import text3d2video.rerun_util as ru

usd_file = Path("/home/jorge/untitled.usdc")

# rr.init("view_animation", spawn=True)
ru.pt3d_setup()
seq = ru.TimeSequence("frames")

stage = Usd.Stage.Open(str(usd_file))

start_time = stage.GetStartTimeCode()
end_time = stage.GetEndTimeCode()
time_step = stage.GetTimeCodesPerSecond()

for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Mesh):
        mesh = UsdGeom.Mesh(prim)

    if prim.IsA(UsdGeom.Camera):
        camera = UsdGeom.Camera(prim)

skel_binding_api = UsdSkel.BindingAPI(mesh)
skel_binding = skel_binding_api.GetSkeleton()

# for frame in range(int(start_time), int(end_time) + 1):
for frame in range(1, 2):
    focal_length = camera.GetFocalLengthAttr().Get(frame)
    horizontal_aperture = camera.GetHorizontalApertureAttr().Get(frame)
    vertical_aperture = camera.GetVerticalApertureAttr().Get(frame)
    horizontal_offset = camera.GetHorizontalApertureOffsetAttr().Get(frame)
    vertical_offset = camera.GetVerticalApertureOffsetAttr().Get(frame)

    xform_cache = UsdGeom.XformCache(time=frame)
    c2w = xform_cache.GetLocalToWorldTransform(camera.GetPrim())

    translation = c2w.ExtractTranslation()
    rotation = c2w.ExtractRotationMatrix()

    translation = Tensor(translation)
    rotation = Tensor(rotation)

    cam = FoVPerspectiveCameras(
        R=rotation.unsqueeze(0), T=translation.unsqueeze(0), fov=60
    )

    if skel_binding:
        skel_qry = UsdSkel.SkinningQuery(mesh)
        if skel_qry.HasDeformedPoints():
            print("lol")

    points = mesh.GetPointsAttr().Get(frame)
    face_vert_indices = mesh.GetFaceVertexIndicesAttr().Get()
    face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()

    vertices = Tensor(points)

    # faces = []
    # idx_offset = 0
    # for count in face_vertex_counts:
    #     face = face_vert_indices[idx_offset : idx_offset + count]
    #     if len(face) == 3:
    #         faces.append(face)
    #     idx_offset += count
    # faces = Tensor(faces)

    # pt3d_mesh = Meshes(verts=[vertices], faces=[faces])

    # rr.log("mesh", ru.pt3d_mesh(pt3d_mesh))

    rr.log("verts", rr.Points3D(vertices))

    # ru.log_pt3d_fov_camera("cam", cam)
    seq.step()
