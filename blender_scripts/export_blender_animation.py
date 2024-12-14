import json
import os
import shutil
from pathlib import Path

import bpy
import numpy as np
from bpy.types import Camera
from mathutils import Matrix

# Get the frame range
scene = bpy.context.scene
start_frame = scene.frame_start
end_frame = scene.frame_end

# find camera
camera: Camera = None
for obj in bpy.data.objects:
    if obj.type == "CAMERA":
        camera = obj
        break

folder = Path("/home/jorge/animation")
if folder.exists():
    shutil.rmtree(folder)
folder.mkdir(parents=True)

cameras_json = folder / "cameras.json"

cam_poses = []
for frame_i in range(start_frame, end_frame + 1):
    # Set the current frame
    scene.frame_set(frame_i)

    # save camera transform
    c2w: Matrix = camera.matrix_world
    translation = np.array(c2w.translation)
    rotation = np.array(c2w.to_3x3())
    cam_pose = {
        "id": frame_i,
        "translation": translation.tolist(),
        "rotation": rotation.flatten().tolist(),
    }
    cam_poses.append(cam_pose)

    # export obj
    filepath = str(folder / f"frame_{frame_i}.obj")
    bpy.ops.wm.obj_export(filepath=filepath, export_materials=False)

# save camera poses
with open(cameras_json, "w") as f:
    json.dump(cam_poses, f, indent=4)
