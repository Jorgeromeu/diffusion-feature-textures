import click
import rerun as rr
import rerun.blueprint as rrb
import torch
from pxr import Usd, UsdGeom
from pytorch3d.renderer import FoVPerspectiveCameras, join_cameras_as_batch
from pytorch3d.structures import Meshes, join_meshes_as_batch
from torch import Tensor
from tqdm import tqdm

import text3d2video.rerun_util as ru
import wandb
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.camera_utils import focal_length_to_fov
from text3d2video.coord_utils import (
    BLENDER_CAM_TO_PT3D_CAM,
    BLENDER_WORLD_TO_PT3D_WORLD,
    apply_transform_homogeneous,
    assemble_transform_srt,
    decompose_transform_srt,
)
from text3d2video.rendering import render_depth_map
from text3d2video.usd_utils import triangulate_usd_mesh, usd_uvs_to_pt3d_uvs
from text3d2video.util import ordered_sample

BLENDER_WORLD_CONVENTION = rr.ViewCoordinates.RFU
BLENDER_CAM_CONVENTION = rr.ViewCoordinates.RUB


def create_animation_from_usd(
    artifact_name: str, usd_path: str, rerun=False, render_res=100, n_frames: int = 100
) -> AnimationArtifact:
    # Blender to Pt3d convention_conversion

    # create animation
    anim = AnimationArtifact.create_empty_artifact(artifact_name)

    if rerun:
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

    # use blender world
    rr.log("/", BLENDER_WORLD_CONVENTION, static=True)

    # open usd file
    stage = Usd.Stage.Open(usd_path)

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
    anim.write_uv_data(verts_uvs, faces_uvs)

    # iterate over frames
    start_time = int(stage.GetStartTimeCode())
    end_time = int(stage.GetEndTimeCode())
    frame_indices = list(range(start_time, end_time + 1))

    if n_frames is not None:
        frame_indices = ordered_sample(frame_indices, n_frames)

    cams_pt3d = []
    meshes_pt3d = []

    for i, frame in enumerate(tqdm(frame_indices)):
        xform_cache = UsdGeom.XformCache(time=frame)

        # mesh to world transform
        m2w = xform_cache.GetLocalToWorldTransform(mesh.GetPrim())
        m2w = Tensor(m2w).T

        # mesh world coords at frame
        verts = Tensor(mesh.GetPointsAttr().Get(frame)).float()
        verts_world = apply_transform_homogeneous(verts, m2w)

        # camera extrinsics
        c2w = xform_cache.GetLocalToWorldTransform(camera.GetPrim())
        c2w = Tensor(c2w).T
        cam_t, _, cam_r = decompose_transform_srt(c2w)

        # camera intrinsics
        focal_length = camera.GetFocalLengthAttr().Get()
        height = camera.GetVerticalApertureAttr().Get()

        # log camera
        if rerun:
            scaling_factor = render_res / height
            height_px = render_res
            width_px = render_res
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

        # log mesh
        if rerun:
            mesh_pt3d = Meshes(verts=[verts_world], faces=[triangle_indices])
            rr.log("mesh", ru.pt3d_mesh(mesh_pt3d))

        # mesh world coordinates in pt3d space at frame
        verts_world_pt3d = verts_world @ BLENDER_WORLD_TO_PT3D_WORLD.T
        mesh_pt3d = Meshes(verts=[verts_world_pt3d], faces=[triangle_indices])

        # c2w in pt3d space
        cam_r_pt3d = BLENDER_WORLD_TO_PT3D_WORLD @ cam_r @ BLENDER_CAM_TO_PT3D_CAM
        cam_t_pt3d = BLENDER_WORLD_TO_PT3D_WORLD @ cam_t
        c2w_pt3d = assemble_transform_srt(cam_t_pt3d, torch.ones(3), cam_r_pt3d)

        # construct pt3d camera
        w2c_pt3d = c2w_pt3d.inverse()
        t_w2c, _, r_w2c = decompose_transform_srt(w2c_pt3d)

        fov = focal_length_to_fov(focal_length, height)
        cam_pt3d = FoVPerspectiveCameras(
            R=r_w2c.T.unsqueeze(0), T=t_w2c.unsqueeze(0), fov=fov
        )

        # log render
        if rerun:
            render = render_depth_map(
                mesh_pt3d.cuda(), cam_pt3d.cuda(), resolution=render_res
            )[0]
            rr.log("camera/render", rr.Image(render))

        cams_pt3d.append(cam_pt3d)
        meshes_pt3d.append(mesh_pt3d)
        seq.step()

    cams_pt3d = join_cameras_as_batch(cams_pt3d)
    meshes_pt3d = join_meshes_as_batch(meshes_pt3d)

    # write animation
    anim.write_frames(cams_pt3d, meshes_pt3d)

    return anim


@click.command()
@click.argument("artifact_name", type=str)
@click.argument("usd_file", type=click.Path(exists=True))
@click.option(
    "--n_frames",
    type=int,
    default=None,
    required=False,
    help="If provided, evenly sample animation frames",
)
@click.option(
    "--rerun",
    required=False,
    is_flag=True,
    help="If set, visualize animation in rerun.io",
)
def cli(artifact_name: str, filename: str, n_frames: int, rerun: bool):
    wandb.init(project="diffusion-3D-features", job_type="log_artifact")

    artifact = create_animation_from_usd(
        artifact_name, filename, n_frames=n_frames, rerun=rerun
    )

    artifact.log_if_enabled()


if __name__ == "__main__":
    cli()
