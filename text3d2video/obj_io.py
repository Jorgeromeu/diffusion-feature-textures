from typing import Optional

from iopath.common.file_io import PathManager
from pytorch3d.common.datatypes import Device
from pytorch3d.io import load_obj
from pytorch3d.renderer import TexturesUV
from pytorch3d.structures import Meshes, join_meshes_as_batch


def load_objs_as_meshes(
    files: list,
    device: Optional[Device] = None,
    load_textures: bool = True,
    texture_wrap: Optional[str] = "repeat",
    path_manager: Optional[PathManager] = None,
):
    """
    Load meshes from a list of .obj files using the load_obj function, and
    return them as a Meshes object. This only works for meshes which have a
    single texture image for the whole mesh. See the load_obj function for more
    details. material_colors and normals are not stored.

    Args:
        files: A list of file-like objects (with methods read, readline, tell,
            and seek), pathlib paths or strings containing file names.
        device: Desired device of returned Meshes. Default:
            uses the current device for the default tensor type.
        load_textures: Boolean indicating whether material files are loaded
        create_texture_atlas, texture_atlas_size, texture_wrap: as for load_obj.
        path_manager: optionally a PathManager object to interpret paths.

    Returns:
        New Meshes object.
    """
    mesh_list = []
    for f_obj in files:
        verts, faces, aux = load_obj(
            f_obj,
            load_textures=load_textures,
            texture_wrap=texture_wrap,
            path_manager=path_manager,
        )

        # TexturesUV type
        tex_maps = aux.texture_images

        if tex_maps is not None and len(tex_maps) > 0:
            verts_uvs = aux.verts_uvs.to(device)  # (V, 2)
            faces_uvs = faces.textures_idx.to(device)  # (F, 3)
            image = list(tex_maps.values())[0].to(device)[None]
            tex = TexturesUV(
                verts_uvs=[verts_uvs], faces_uvs=[faces_uvs], maps=image
            )
        else:
            tex = None
            # use empty texture
            # tex_map = tex.get
            pass

        mesh = Meshes(
            verts=[verts.to(device)], faces=[faces.verts_idx.to(device)], textures=tex
        )
        mesh_list.append(mesh)

    if len(mesh_list) == 1:
        return mesh_list[0]

    return join_meshes_as_batch(mesh_list)
