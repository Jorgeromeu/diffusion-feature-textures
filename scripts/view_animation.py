import json
from pathlib import Path
from typing import List

import rerun as rr
import torch
from pytorch3d.io import load_objs_as_meshes

import text3d2video.rerun_util as ru
from text3d2video.util import ordered_sample


def parse_json_matrix(json_lst: List[str]):
    float_lst = [float(x) for x in json_lst]
    return torch.tensor(float_lst)


animation_path = Path("/home/jorge/animation")
cameras_path = animation_path / "cameras.json"

with open(cameras_path, "r") as f:
    cameras = json.load(f)

rr.init("view_animation", spawn=True)
ru.pt3d_setup()
seq = ru.TimeSequence("frames")

frame_ids = [int(camera["id"]) for camera in cameras]
frame_ids = ordered_sample(frame_ids, 30)

mesh_paths = [animation_path / f"frame_{id}.obj" for id in frame_ids]
meshes = load_objs_as_meshes([str(p) for p in mesh_paths])

for i, frame in enumerate(frame_ids):
    mesh = meshes[i]
    camera = cameras[i]

    tvec = parse_json_matrix(camera["translation"])
    rotation = parse_json_matrix(camera["rotation"]).reshape(3, 3)

    rr.log("mesh", ru.pt3d_mesh(mesh))

    rr.log("camera", rr.Pinhole(height=100, width=100, focal_length=100))
    rr.log("camera", rr.Transform3D(translation=tvec, mat3x3=rotation))

    seq.step()

# for camera in data:
#     id = int(camera["id"])
#     mesh_path = animation_path / f"frame_{id}.obj"
#     # print(mesh_path)
#     # tvec = parse_json_matrix(camera["translation"])
#     # print(id)

#     # rr.log("point", rr.Points3D(tvec.unsqueeze(0)))
