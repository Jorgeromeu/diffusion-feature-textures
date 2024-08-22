import numpy as np
import rerun as rr
from pytorch3d.renderer import FoVPerspectiveCameras
from pytorch3d.structures import Meshes
from pytorch3d.transforms import Transform3d

PT3D_ViewCoords = rr.ViewCoordinates.LUF

def pt3d_mesh(meshes: Meshes, batch_idx=0):
    # extract verts and faces from idx-th mesh in batch
    verts = meshes.verts_list()[batch_idx].cpu()
    faces = meshes.faces_list()[batch_idx].cpu()
    vertex_normals = meshes.verts_normals_list()[batch_idx].cpu()

    # TODO figure out hw to render textures...
    tex = meshes.textures.maps_padded()[batch_idx].cpu().numpy()
    uv = meshes.textures.verts_uvs_padded()[0].cpu().numpy()

    return rr.Mesh3D(
        vertex_positions=verts,
        triangle_indices=faces,
        vertex_normals=vertex_normals,
    )

def pt3d_FovCamera(cameras: FoVPerspectiveCameras, batch_idx=0):
    # TODO figure out how to get size from raster settings
    sensor_size = 300
    fov = cameras[batch_idx].fov.item()
    focal_length = int(sensor_size / (2 * np.tan(fov * np.pi / 360)))

    return rr.Pinhole(height=sensor_size, width=sensor_size, focal_length=focal_length)

def pt3d_transform(transforms: Transform3d, batch_idx=0):
    matrix = transforms.get_matrix()

    translation = matrix[batch_idx, 3, 0:3].cpu()
    rotation = matrix[batch_idx, 0:3, 0:3].cpu().inverse()

    return rr.Transform3D(
        translation=translation,
        mat3x3=rotation
    )
