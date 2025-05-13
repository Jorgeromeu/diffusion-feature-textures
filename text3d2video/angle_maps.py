import torch
from einops import rearrange
from pytorch3d.ops import interpolate_face_attributes

from text3d2video.rendering import make_mesh_rasterizer


def get_world_rays(cameras, resolution: int = 512):
    device = cameras.device

    # Create screen space grid in NDC [-1, 1]
    xs = torch.linspace(-1, 1, resolution, device=device)
    ys = torch.linspace(-1, 1, resolution, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
    pixels_ndc = torch.stack(
        [grid_x, grid_y, -torch.ones_like(grid_x)], dim=-1
    )  # (H, W, 3)

    # Convert to homogeneous coordinates (x, y, z, 1)
    pixels_h = torch.cat(
        [pixels_ndc, torch.ones_like(pixels_ndc[..., :1])], dim=-1
    )  # (H, W, 4)

    # Expand for batch
    pixels_h = pixels_h.unsqueeze(0)  # (1, H, W, 4)

    # Transform from view to world space
    view_to_world = (
        cameras.get_world_to_view_transform().inverse().get_matrix()
    )  # (1, 4, 4)
    pixels_world = torch.matmul(pixels_h, view_to_world.transpose(1, 2))  # (1, H, W, 4)

    # Compute world rays: pixel_world - camera_center
    rays = pixels_world[..., :3] - cameras.get_camera_center().view(
        1, 1, 1, 3
    )  # (1, H, W, 3)
    rays = torch.nn.functional.normalize(rays, dim=-1)
    return rays[0]  # (H, W, 3)


def render_view_angle_map(cam, mesh, resolution=512):
    # get vert nromals
    vert_normals = mesh.verts_normals_list()[0]
    faces = mesh.faces_list()[0]
    face_vert_normals = vert_normals[faces]

    # get normal at pixels
    rasterizer = make_mesh_rasterizer(resolution=resolution)
    frags = rasterizer(mesh, cameras=cam)
    pixel_normals = interpolate_face_attributes(
        frags.pix_to_face, frags.bary_coords, face_vert_normals
    )

    pixel_normals = rearrange(pixel_normals, "1 h w 1 c -> h w c")

    # get view-dirs at pixels
    view_dirs = get_world_rays(cam, resolution=resolution)  # (H, W, 3)

    # get view-angles
    cos_theta = (view_dirs * pixel_normals).sum(dim=-1).clamp(-1.0, 1.0)  # (H, W)
    theta = torch.acos(cos_theta)  # (H, W)

    # set bg to 0
    bg_mask = (frags.zbuf < 0)[0, :, :, 0]
    theta[bg_mask] = 0

    return theta


def render_view_angle_quality_map(cam, mesh, resolution=512):
    angle = render_view_angle_map(cam, mesh, resolution=resolution)
    quality = angle / torch.pi
    return 1 - quality
