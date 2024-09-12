import time
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
from text3d2video.util import project_vertices_to_features, multiview_cameras, random_solid_color_img
from text3d2video.visualization import RgbPcaUtil
from PIL import Image

def compute_3d_diffusion_features(
        pipe: StableDiffusionControlNetPipeline,
        mesh: Meshes,
        prompt: str = 'Deadpool',
        n_views=9,
        resolution=512,
        device='cpu',
        log_pca_features=False
) -> torch.Tensor:
    
    """
    Compute Diffusion 3D Features for a given mesh, and represent them as vertex features.
    :param pipe: Depth 2 Image Diffusion pipeline to extract features from
    :param mesh: Pytorch3D Meshes object representing the mesh
    :param prompt: Prompt to generate images from
    :param n_views: Number of views to render depth maps from
    :param resolution: Resolution of depth maps, and generated images
    :param device: Device to run the computation 
    :return: Vertex features representing the diffusion features
    """

    # manual timesteps
    rr_seq = ru.TimeSequence("steps")
    
    # log original mesh
    rr.log('mesh', ru.pt3d_mesh(mesh))
    
    rr_seq.step()
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

    depth_maps = normalize_depth_map(fragments.zbuf)
    depth_imgs = [TF.to_pil_image(depth_maps[i, :, :, 0]) 
                  for i in range(n_views)]

    rr_seq.step()

    # log depth images
    for i in range(n_views):
        rr.log(f'cam_{i}', rr.Image(depth_imgs[i]))

    # setup feature extractor
    feature_extractor = SDFeatureExtractor(pipe)

    # Generate images
    prompts = [prompt] * n_views
    generted_ims = depth2img(pipe, prompts, depth_imgs, num_inference_steps=30)

    rr_seq.step()

    # log generated images
    for i in range(n_views):
        rr.log(f'cam_{i}', rr.Image(generted_ims[i]))

    # extract features
    # size = n_views
    extracted_features = feature_extractor.get_feature(level=2, timestep=20)

    feature_maps = []
    for i in range(n_views):
        feature_map = torch.Tensor(extracted_features[i])
        feature_maps.append(feature_map)

    rr_seq.step()

    # log feature maps
    rr.log(f'feature_{i}', ru.feature_map(feature_map.cpu().numpy()))

    # initialize empty D-dimensional vertex features
    feature_dim = feature_maps[0].shape[0]
    vertex_features = torch.zeros(mesh.num_verts_per_mesh()[0], feature_dim)
    vertex_feature_count = torch.zeros(mesh.num_verts_per_mesh()[0])

    for i in range(n_views):
        # project view features to vertices
        feature_map = feature_maps[i]
        view_vertex_features = project_vertices_to_features(
            mesh,
            cameras,
            feature_map,
            batch_idx=i
        ).cpu()

        # indices of vertices with nonzero view features
        nonzero_indices = torch.where(torch.any(view_vertex_features != 0, dim=1))[0]

        # update vertex features
        vertex_features += view_vertex_features
        vertex_feature_count[nonzero_indices] += 1

    if log_pca_features:

        # fit PCA matrix
        pca = RgbPcaUtil(feature_dim)
        pca.fit(vertex_features)

        # apply PCA matrix
        reduced_features = pca.features_to_rgb(vertex_features)

        # log each view's dimensionality-reduced feature map
        rr_seq.step()
        for i in range(n_views):
            feature_map = feature_maps[i]
            feature_map_resolution = feature_map.shape[1]
            fmap_rgb = pca.feature_map_to_rgb(feature_map)
            fmap_rgb = TF.to_pil_image(fmap_rgb)

            # log feature map (and camera because of new resolution)
            ru.log_pt3d_FovCamrea(f'cam_{i}', cameras, batch_idx=i, res=feature_map_resolution)
            rr.log(f'cam_{i}', rr.Image(fmap_rgb))

        # log reduced features on mesh
        rr_seq.step()
        rr.log('mesh', ru.pt3d_mesh(mesh, vertex_colors=reduced_features))

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
        prompt=prompt,
        log_pca_features=True
    )

    # save computed vert features
    torch.save(vert_features, f'outs/{mesh_name}_vert_features.pt')

    # reduce features
    # reduced_features = reduce_features(vert_features)

    # # log reduced features on mesh
    # rr.log('mesh', ru.pt3d_mesh(mesh, vertex_colors=reduced_features))
