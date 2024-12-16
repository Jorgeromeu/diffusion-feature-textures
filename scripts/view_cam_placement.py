import time

import rerun as rr
import rerun.blueprint as rrb
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import CamerasBase
from pytorch3d.structures import Meshes

import text3d2video.rerun_util as ru
from text3d2video.camera_placement import turntable_cameras
from text3d2video.camera_utils import fov_to_focal_length
from text3d2video.coord_utils import decompose_transform_srt

PT3D_camera_convention = rr.ViewCoordinates.LUF
PT3D_world_convention = rr.ViewCoordinates.LUF

blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(name="3D"),
    ),
    collapse_panels=True,
)
rr.init("view_cam_placement")
rr.serve()
rr.send_blueprint(blueprint)
seq = ru.TimeSequence("animation")

# PT3D world convention
rr.log("/", PT3D_world_convention, static=True)

resolution = 300

mesh = load_objs_as_meshes(["data/meshes/mixamo-human.obj"])

rr.log("mesh", ru.pt3d_mesh(mesh))


def mesh_views(mesh: Meshes, cameras: CamerasBase, resolution=100):
    pass


cameras = turntable_cameras(20, 2)

for i, cam in enumerate(cameras):
    w2c = cam.get_world_to_view_transform().get_matrix()[0].T
    c2w = w2c.inverse()

    t, _, r = decompose_transform_srt(c2w)
    t = t.cpu()
    r = r.cpu()

    rr.log(
        f"camera_{i}",
        rr.Pinhole(
            height=resolution,
            width=resolution,
            focal_length=fov_to_focal_length(cam.fov.cpu().item(), resolution),
            camera_xyz=PT3D_camera_convention,
        ),
    )
    rr.log(f"camera_{i}", rr.Transform3D(translation=t, mat3x3=r))

    seq.step()

time.sleep(20)

# mesh: Meshes
# camera: FoVPerspectiveCameras

# verts = mesh.verts_list()[0].cpu()
# faces = mesh.faces_list()[0].cpu()
# normals = meshes.verts_normals_list()[0].cpu()

# rr.log(
#     "mesh",
#     rr.Mesh3D(
#         vertex_positions=verts, triangle_indices=faces, vertex_normals=normals
#     ),
# )


# render = render_depth_map(mesh.cuda(), camera.cuda(), resolution=resolution)[0]
# rr.log("camera/render", rr.Image(render))

# seq.step()
