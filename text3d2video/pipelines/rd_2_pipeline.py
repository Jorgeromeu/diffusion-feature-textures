import torch
import torchvision.transforms.functional as TF
from attr import dataclass
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
)
from pytorch3d.structures import Meshes
from tqdm import tqdm

from text3d2video.attn_processors.extraction_injection_attn import (
    ExtractionInjectionAttn,
)
from text3d2video.noise_initialization import NoiseInitializer
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingPipeline,
)
from text3d2video.rendering import (
    TextureShader,
    make_repeated_uv_texture,
    precompute_rast_fragments,
    render_depth_map,
)
from text3d2video.sd_feature_extraction import AttnLayerId
from text3d2video.util import dict_filter, dict_map
from text3d2video.utilities.tensor_writing import FeatureLogger


@dataclass
class ReposableDiffusionConfig:
    do_pre_attn_injection: bool
    do_post_attn_injection: bool
    feature_blend_alpha: float
    attend_to_self_kv: bool
    mean_features_weight: float
    chunk_size: int
    num_inference_steps: int
    guidance_scale: float
    controlnet_conditioning_scale: float
    module_paths: list[str]


class ReposableDiffusion2Pipeline(GenerativeRenderingPipeline):
    conf: ReposableDiffusionConfig
    attn_processor: ExtractionInjectionAttn

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        tgt_meshes: Meshes,
        tgt_cams: FoVPerspectiveCameras,
        verts_uvs: torch.Tensor,
        faces_uvs: torch.Tensor,
        reposable_diffusion_config: ReposableDiffusionConfig,
        noise_initializer: NoiseInitializer,
        extracted_feats: FeatureLogger,
        generator=None,
    ):
        # setup configs for use throughout pipeline
        self.conf = reposable_diffusion_config
        self.noise_initializer = noise_initializer

        # setup attn processor
        self.attn_processor = ExtractionInjectionAttn(
            self.unet,
            do_spatial_post_attn_extraction=self.conf.do_post_attn_injection,
            do_kv_extraction=self.conf.do_pre_attn_injection,
            also_attend_to_self=self.conf.attend_to_self_kv,
            feature_blend_alpha=self.conf.feature_blend_alpha,
            kv_extraction_paths=self.conf.module_paths,
            spatial_post_attn_extraction_paths=self.conf.module_paths,
        )
        self.unet.set_attn_processor(self.attn_processor)

        # configure scheduler
        self.scheduler.set_timesteps(self.conf.num_inference_steps)

        n_frames = len(tgt_meshes)
        frame_indices = torch.arange(n_frames)

        # render all depth maps
        depth_maps = render_depth_map(tgt_meshes, tgt_cams, 512)

        layers = [
            AttnLayerId.parse_module_path(path) for path in self.conf.module_paths
        ]
        screen_resolutions = list(
            set([layer.layer_resolution(self.unet) for layer in layers])
        )

        raster_resolutions = [10, 20, 32, 64]

        layer_resolution_indices = {
            layer.module_path(): screen_resolutions.index(
                layer.layer_resolution(self.unet)
            )
            for layer in layers
        }

        fragments = precompute_rast_fragments(tgt_cams, tgt_meshes, raster_resolutions)

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt] * n_frames)

        # initial noise
        latents = self.prepare_latents(
            tgt_meshes, tgt_cams, verts_uvs, faces_uvs, generator
        )

        t_weight = 0.8
        max_t_i = self.conf.num_inference_steps * t_weight

        # denoising loop
        for t_i, t in enumerate(tqdm(self.scheduler.timesteps)):
            self.attn_processor.set_cur_timestep(t)

            def read_features(t, name):
                return {
                    layer: extracted_feats.read(layer, int(t), name).to(
                        device=self.device, dtype=self.dtype
                    )
                    for layer in self.conf.module_paths
                }

            kvs_cond = read_features(t, "kvs_cond")
            kvs_uncond = read_features(t, "kvs_uncond")
            textures_cond = read_features(t, "tex_cond")
            textures_uncond = read_features(t, "tex_uncond")

            if not self.conf.do_post_attn_injection:
                textures_cond = {}
                textures_uncond = {}

            if not self.conf.do_pre_attn_injection:
                kvs_cond = {}
                kvs_uncond = {}

            def filter(layer, _):
                layers = self.conf.module_paths
                layers_weight = 0.8
                max_layer_index = int(len(layers) * layers_weight)
                return layer in self.conf.module_paths[0:max_layer_index]

            textures_cond = dict_filter(textures_cond, filter)
            textures_uncond = dict_filter(textures_uncond, filter)

            # Denoise target frames in chunks with injection
            noise_preds = []
            for chunk_frame_indices in torch.split(frame_indices, self.conf.chunk_size):

                def render_chunk_frames(layer, texture):
                    shader = TextureShader()
                    tex = make_repeated_uv_texture(
                        texture,
                        faces_uvs,
                        verts_uvs,
                        sampling_mode="nearest",
                    )

                    res_idx = layer_resolution_indices[layer]
                    resolution = screen_resolutions[res_idx]

                    renders = []
                    for frame_i in chunk_frame_indices.tolist():
                        mesh = tgt_meshes[frame_i]
                        frags = fragments[res_idx][frame_i]
                        mesh.textures = tex
                        render = shader(frags, mesh)[0]
                        render = TF.resize(
                            render,
                            (resolution, resolution),
                            interpolation=TF.InterpolationMode.NEAREST,
                        )

                        renders.append(render)
                    renders = torch.stack(renders)
                    return renders

                chunk_rendered_cond = dict_map(textures_cond, render_chunk_frames)
                chunk_rendered_uncond = dict_map(textures_uncond, render_chunk_frames)

                if t_i > max_t_i:
                    chunk_rendered_cond = {}
                    chunk_rendered_uncond = {}

                chunk_noise = self.model_forward_injection(
                    latents[chunk_frame_indices],
                    cond_embeddings[chunk_frame_indices],
                    uncond_embeddings[chunk_frame_indices],
                    [depth_maps[i] for i in chunk_frame_indices.tolist()],
                    t,
                    kvs_cond,
                    kvs_uncond,
                    chunk_rendered_cond,
                    chunk_rendered_uncond,
                )

                noise_preds.append(chunk_noise)

            # Denoise all
            noise_preds = torch.cat(noise_preds)
            latents = self.scheduler.step(
                noise_preds, t, latents, generator=generator
            ).prev_sample

        # decode latents in chunks
        decoded_imgs = []
        for chunk_frame_indices in torch.split(frame_indices, self.conf.chunk_size):
            chunk_latents = latents[chunk_frame_indices]
            chunk_images = self.decode_latents(chunk_latents, generator)
            decoded_imgs.extend(chunk_images)

        return decoded_imgs
