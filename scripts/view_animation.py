import rerun as rr
import rerun.blueprint as rrb
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Meshes

import text3d2video.rerun_util as ru
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.coord_utils import decompose_transform_srt
from text3d2video.rendering import render_depth_map

animation = AnimationArtifact.from_wandb_artifact_tag("human_rotation_full:latest")
animation = AnimationArtifact.from_wandb_artifact_tag("booty:latest")

frame_indices = animation.frame_indices()
cameras, meshes = animation.load_frames()

PT3D_camera_convention = rr.ViewCoordinates.LUF
PT3D_world_convention = rr.ViewCoordinates.LUF


blueprint = rrb.Blueprint(
    rrb.Horizontal(
        rrb.Spatial3DView(name="3D"),
        rrb.Spatial2DView(name="2D", contents=["/camera/render"], origin=["camera"]),
    ),
    collapse_panels=True,
)
rr.init("view_animation_pt3d")
rr.serve()
rr.send_blueprint(blueprint)
seq = ru.TimeSequence("animation")

rr.log("/", PT3D_world_convention, static=True)

resolution = 300

for mesh, camera in zip(meshes, cameras):
    mesh: Meshes
    camera: FoVPerspectiveCameras

    verts = mesh.verts_list()[0].cpu()
    faces = mesh.faces_list()[0].cpu()
    normals = meshes.verts_normals_list()[0].cpu()

    rr.log(
        "mesh",
        rr.Mesh3D(
            vertex_positions=verts, triangle_indices=faces, vertex_normals=normals
        ),
    )

    w2c = camera.get_world_to_view_transform().get_matrix()[0].T
    c2w = w2c.inverse()

    t, _, r = decompose_transform_srt(c2w)
    t = t.cpu()
    r = r.cpu()

    rr.log(
        "camera",
        rr.Pinhole(
            height=resolution,
            width=resolution,
            focal_length=ru.fov_to_focal_length(camera.fov.cpu().item(), resolution),
            camera_xyz=PT3D_camera_convention,
        ),
    )
    rr.log("camera", rr.Transform3D(translation=t, mat3x3=r))

    render = render_depth_map(mesh.cuda(), camera.cuda(), resolution=resolution)[0]
    rr.log("camera/render", rr.Image(render))

    seq.step()
