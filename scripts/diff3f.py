from einops import repeat
import torch
from diffusers import StableDiffusionControlNetPipeline
from pytorch3d.structures import Meshes
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import look_at_view_transform, RasterizationSettings, MeshRasterizer
import rerun as rr
import rerun.blueprint as rrb
from text3d2video.diffusion import depth2img, depth2img_pipe
from text3d2video.rendering import normalize_depth_map
import text3d2video.rerun_util as ru
import torchvision.transforms.functional as TF
import faiss
from text3d2video.sd_feature_extraction import SDFeatureExtractor
from text3d2video.util import feature_per_vertex, multiview_cameras, random_solid_color_img
from text3d2video.visualization import reduce_feature_map, reduce_features

def compute_3d_diffusion_features(
        pipe: StableDiffusionControlNetPipeline,
        mesh: Meshes,
        prompt: str = 'Deadpool',
        n_views=10,
        resolution=512,
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
    depth_maps = normalize_depth_map(fragments.zbuf)
    depth_imgs = [TF.to_pil_image(depth_maps[i, :, :, 0]) 
                  for i in range(n_views)]
    
    for map in depth_imgs:
        map.save('outs/depth_map.png')

    # log depth maps
    for i in range(n_views):
        rr.log(f'cam_{i}', rr.Image(depth_imgs[i]))

    feature_extractor = SDFeatureExtractor(pipe)

    # Generate images
    prompts = [prompt] * n_views
    generted_ims = depth2img(pipe, prompts, depth_imgs, num_inference_steps=30)
    
    # log generated images
    for i in range(n_views):
        rr.log(f'cam_{i}', rr.Image(generted_ims[i]))

    # extract features
    extracted_features = feature_extractor.get_feature(level=1, timestep=20)

    feature_maps = []
    for i in range(n_views):
        fmap = torch.Tensor(extracted_features[i*2])
        feature_maps.append(fmap)

    feature_dim = feature_maps[0].shape[0]

    # initialize empty vertex features
    vertex_features = torch.zeros(mesh.num_verts_per_mesh()[0], feature_dim)
    vertex_feature_count = torch.zeros(mesh.num_verts_per_mesh()[0])

    for i in range(n_views):
        # project view features to vertices
        feature_map = feature_maps[i]
        view_vertex_features = feature_per_vertex(mesh, cameras, feature_map, batch_idx=i).cpu()

        # indices of vertices with nonzero view features
        nonzero_indices = torch.where(torch.any(view_vertex_features != 0, dim=1))[0]

        # update vertex features
        vertex_features += view_vertex_features
        vertex_feature_count[nonzero_indices] += 1
    
    return vertex_features

if __name__ == "__main__":

    mesh_name = 'mixamo-human'
    prompt = 'Deadpool'
    mesh = f'data/{mesh_name}.obj'
    n_views = 15

    # init 3D logging
    rr.init('diff3f')
    rr.serve()
    ru.pt3d_setup()

    # log blueprint
    blueprint = rrb.Blueprint(rrb.Spatial3DView())
    rr.send_blueprint(blueprint)

    # load mesh 
    device = 'cuda:0'
    mesh: Meshes = load_objs_as_meshes([mesh], device=device)

    # load pipeline
    pipe = depth2img_pipe(device=device)

    # compute diffusion features
    vert_features = compute_3d_diffusion_features(
        pipe,
        mesh,
        device=device,
        n_views=n_views,
        prompt=prompt
    )

    torch.save(vert_features, f'outs/{mesh_name}_vert_features.pt')

    reduced_features = reduce_features(vert_features)

    rr.log('mesh', ru.pt3d_mesh(mesh, vertex_colors=reduced_features))
