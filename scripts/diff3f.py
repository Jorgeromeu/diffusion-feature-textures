from einops import repeat
import torch
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import look_at_view_transform, RasterizationSettings, MeshRasterizer
import rerun as rr
import rerun.blueprint as rrb
from text3d2video.rendering import normalize_depth_map
import text3d2video.rerun_util as ru
import torchvision.transforms.functional as TF

from text3d2video.util import feature_per_vertex, multiview_cameras, random_solid_color_img

def compute_3d_diffusion_features(
        mesh: Meshes,
        n_views=10,
        resolution=100,
        device='cpu'
):

    # log original mesh
    rr.log('mesh', ru.pt3d_mesh(mesh))

    # generate cameras
    cameras = multiview_cameras(mesh, n_views, device=device)
    n_views = len(cameras)

    # log cameras
    for i in range(n_views):
        ru.log_pt3d_FovCamrea(f'cam_{i}', cameras, batch_idx=i, res=resolution)

    # render depth maps
    raster_settings = RasterizationSettings(
        image_size=resolution,
        faces_per_pixel=1
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    batch_mesh = mesh.extend(n_views)
    fragments = rasterizer(batch_mesh)

    # get depth maps
    depth_maps = normalize_depth_map(fragments.zbuf.cpu().numpy())
    depth_imgs = [TF.to_pil_image(depth_maps[i, :, :, 0]) for i in range(n_views)]

    # log depth maps
    for i in range(n_views):
        rr.log(f'cam_{i}', rr.Image(depth_imgs[i]))

    # TODO replace with true generated images
    generated_ims = [random_solid_color_img() for _ in range(n_views)]
    generated_ims_pil = [TF.to_pil_image(im) for im in generated_ims]

    # log generated images
    for i in range(n_views):
        rr.log(f'cam_{i}', rr.Image(generated_ims_pil[i]))

    vertex_features = torch.zeros(mesh.num_verts_per_mesh()[0], 3)
    vertex_feature_count = torch.zeros(mesh.num_verts_per_mesh()[0])

    for i in range(n_views):
        view_vertex_features = feature_per_vertex(mesh, cameras, generated_ims[i], batch_idx=i).cpu()

        # indices of vertices with nonzero view features
        nonzero_indices = torch.where(torch.any(view_vertex_features != 0, dim=1))[0]

        vertex_features += view_vertex_features
        vertex_feature_count[nonzero_indices] += 1

        # average vertex features
        avg_vertex_features = vertex_features / vertex_feature_count.unsqueeze(1)
        
        # rr.log('mesh', ru.pt3d_mesh(mesh, vertex_colors=view_vertex_features.cpu().numpy()))
        rr.log('mesh', ru.pt3d_mesh(mesh, vertex_colors=avg_vertex_features.cpu().numpy()))


if __name__ == "__main__":

    # init 3D logging
    rr.init(spawn=True, application_id='diff3F')
    ru.pt3d_setup()

    # log blueprint
    blueprint = rrb.Blueprint(rrb.Spatial3DView())
    rr.send_blueprint(blueprint)

    # load mesh 
    device = 'cuda:0'
    mesh: Meshes = load_objs_as_meshes(['data/max-planck.obj'], device=device)

    compute_3d_diffusion_features(mesh, device=device)