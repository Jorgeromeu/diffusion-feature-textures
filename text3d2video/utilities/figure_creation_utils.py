from pathlib import Path

from PIL import Image, ImageDraw


def write_image_seq(folder: Path, name, images):
    seq_path = folder / name
    seq_path.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(images):
        im.save(seq_path / f"im_{i:03d}.png")


def add_filmstrip_border_clean(
    img, slit_width=4, slit_height=10, slit_spacing=15, strip_height=20
):
    """Adds top and bottom film strip borders with rectangular slits (cleaner look)."""
    w, h = img.size
    new_h = h + 2 * strip_height
    result = Image.new("RGB", (w, new_h), "black")
    result.paste(img, (0, strip_height))

    draw = ImageDraw.Draw(result)
    for y_offset in [0, new_h - strip_height]:
        for x in range(0, w, slit_spacing):
            x0 = x
            x1 = x + slit_width
            y0 = y_offset + (strip_height - slit_height) // 2
            y1 = y0 + slit_height
            draw.rectangle([x0, y0, x1, y1], fill="white")

    return result


def filmstrip_sequence(
    images, slit_width=4, slit_height=10, slit_spacing=15, strip_height=20, spacing=5
):
    """Returns a single PIL image with input images arranged and styled as a filmstrip."""
    processed = [
        add_filmstrip_border_clean(
            img, slit_width, slit_height, slit_spacing, strip_height
        )
        for img in images
    ]

    widths = [img.width for img in processed]
    heights = [img.height for img in processed]
    total_width = sum(widths) + spacing * (len(processed) - 1)
    max_height = max(heights)

    result = Image.new("RGB", (total_width, max_height), "white")
    x_offset = 0
    for img in processed:
        result.paste(img, (x_offset, 0))
        x_offset += img.width + spacing

    return result


import numpy as np
import trimesh
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.structures import Meshes


def pytorch3d_to_trimesh(mesh: Meshes) -> trimesh.Trimesh:
    verts = mesh.verts_packed().cpu().numpy()
    faces = mesh.faces_packed().cpu().numpy()
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def make_camera_frustum(R, T, scale=0.1):
    """
    Returns a Trimesh pyramid mesh representing the camera frustum.
    Assumes a simple pinhole camera visual (not actual field of view).
    R: (3, 3), T: (3,)
    """
    # Frustum in camera space
    apex = np.array([[0, 0, 0]])
    base = np.array([[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]) * scale
    verts_cam = np.vstack([apex, base])

    # Transform to world space: X_world = R^T * (X_cam - T)
    R = R.T
    verts_world = verts_cam @ R + T

    # Define faces (pyramid)
    faces = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 1], [1, 2, 3], [1, 3, 4]])

    return trimesh.Trimesh(vertices=verts_world, faces=faces, process=False)


def export_scene_with_cameras(
    mesh: Meshes, cameras: PerspectiveCameras, path="scene.glb"
):
    tm_mesh = pytorch3d_to_trimesh(mesh)
    frusta = []

    R = cameras.R.cpu().numpy()
    T = cameras.T.cpu().numpy()
    for r, t in zip(R, T):
        frusta.append(make_camera_frustum(r, t, scale=0.1))

    # Combine all into one scene
    scene = trimesh.Scene()
    scene.add_geometry(tm_mesh, node_name="mesh")
    for i, cam in enumerate(frusta):
        scene.add_geometry(cam, node_name=f"cam_{i}")

    # Export
    scene.export(path)


# Example usage:
# export_scene_with_cameras(mesh, cameras, "scene.glb")
