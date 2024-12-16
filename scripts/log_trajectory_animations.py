from pathlib import Path
from typing import List

from attr import dataclass
from pytorch3d.io import load_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from rerun import Tensor

import text3d2video.wandb_util as wbu
import wandb
from text3d2video.artifacts.anim_artifact import AnimationArtifact
from text3d2video.camera_trajectories import (
    BarrelRoll,
    BarrelRollPartial,
    NamedCameraTrajectory,
)
from text3d2video.mesh_processing import normalize_meshes
from text3d2video.rendering import render_depth_map
from text3d2video.video_util import pil_frames_to_clip


@dataclass
class Mesh:
    name: str
    mesh_normalized: Meshes
    verts_uvs: Tensor
    faces_uvs: Tensor


def log_cam_movement_animations(
    obj_paths: List[str],
    trajectories: List[NamedCameraTrajectory],
    N: int,
):
    # read meshes
    meshes = load_objs_as_meshes(obj_paths)
    meshes_normalized = normalize_meshes(meshes)

    mesh_scenes = []
    for path, mesh in zip(obj_paths, meshes_normalized):
        path = Path(path)
        mesh_name = path.stem

        # read UV data
        _, faces, aux = load_obj(path)
        verts_uvs = aux.verts_uvs
        faces_uvs = faces.textures_idx

        mesh_scenes.append(Mesh(mesh_name, mesh, verts_uvs, faces_uvs))

    # log animations
    for trajectory in trajectories:
        cams = trajectory.cameras(N)

        for mesh_scene in mesh_scenes:
            anim_name = f"{mesh_scene.name}_{trajectory.name}"
            anim = AnimationArtifact.create_empty_artifact(anim_name)

            mesh_frames = mesh_scene.mesh_normalized.extend(N)
            anim.write_frames(cams, mesh_frames)
            anim.write_uv_data(mesh_scene.verts_uvs, mesh_scene.faces_uvs)

            wandb.init(project="diffusion-3D-features", job_type="log_artifact")
            anim.log_if_enabled()

            # log depth clip
            depth_maps = render_depth_map(
                mesh_frames.cuda(), cams.cuda(), chunk_size=10
            )
            clip = pil_frames_to_clip(depth_maps, duration=2)
            wbu.log_moviepy_clip("animation", clip)

            wandb.finish()


if __name__ == "__main__":
    obj_paths = ["data/meshes/cat.obj", "data/meshes/mixamo-human.obj"]
    trajectories: List[NamedCameraTrajectory] = [
        # RotationFull(),
        # RotationPartial(),
        # Rotation90(),
        # OrthographicPan(),
        # FoVZoom(),
        BarrelRoll(),
        BarrelRollPartial(),
    ]
    log_cam_movement_animations(obj_paths, trajectories, N=100)
