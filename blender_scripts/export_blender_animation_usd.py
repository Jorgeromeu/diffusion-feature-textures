import bpy

# Export Baked Animation to USD format

bpy.ops.wm.usd_export(
    filepath="/home/jorge/untitled.usdc",
    export_cameras=True,
    export_animation=True,
    export_uvmaps=True,
    export_normals=True,
    # ensure triangle mesh
    triangulate_meshes=True,
    # bake the animation
    export_armatures=False,
    # disable everything else
    export_materials=False,
    export_hair=False,
    export_textures=False,
    export_lights=False,
    export_custom_properties=False,
    convert_world_material=False,
    # render mode
    evaluation_mode="RENDER",
)
