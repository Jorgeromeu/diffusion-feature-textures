import torch
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
    make_mesh_renderer,
    make_repeated_uv_texture,
    render_depth_map,
)
from text3d2video.sd_feature_extraction import AttnLayerId
from text3d2video.util import dict_map
from text3d2video.utilities.h5_util import dataset_to_tensor
from text3d2video.utilities.tensor_writing import H5logger


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
        logged_features: H5logger,
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

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt] * n_frames)

        # initial noise
        latents = self.prepare_latents(
            tgt_meshes, tgt_cams, verts_uvs, faces_uvs, generator
        )

        # Read Features
        kvs_cond_dsets = {}
        kvs_uncond_dsets = {}
        textures_cond_dsets = {}
        textures_uncond_dsets = {}

        for module in self.conf.module_paths:

            def read_dsets(path):
                dsets = logged_features.read_datasets(path)
                return sorted(dsets, key=lambda dset: dset.attrs["t"])

            read_dsets(f"kvs_cond/{module}")
            kvs_cond_dsets[module] = read_dsets(f"kvs_cond/{module}")

            kvs_uncond_dsets[module] = read_dsets(f"kvs_uncond/{module}")
            textures_cond_dsets[module] = read_dsets(f"qrys_cond/{module}")
            textures_uncond_dsets[module] = read_dsets(f"qrys_uncond/{module}")

        # denoising loop
        for t in tqdm(self.scheduler.timesteps):
            key = next(iter(kvs_cond_dsets.keys()))
            dsets = kvs_cond_dsets[key]
            dset_index = 0
            for i, dset in enumerate(dsets):
                if dset.attrs["t"] == t.item():
                    dset_index = i

            def read_dataset(dsets):
                return dataset_to_tensor(dsets[dset_index]).to(latents)

            def read_dset_kv(layer, dsets):
                return read_dataset(dsets)[0]

            def read_dset_texture(layer, dsets):
                return read_dataset(dsets)

            kvs_cond = dict_map(kvs_cond_dsets, read_dset_kv)
            kvs_uncond = dict_map(kvs_uncond_dsets, read_dset_kv)
            textures_cond = dict_map(textures_cond_dsets, read_dset_texture)
            textures_uncond = dict_map(textures_uncond_dsets, read_dset_texture)

            # set attn processor timestep
            self.attn_processor.set_cur_timestep(t)

            # Denoise target frames in chunks with injection
            noise_preds = []
            for chunk_frame_indices in torch.split(frame_indices, self.conf.chunk_size):

                def render_chunk_frames(layer, texture):
                    chunk_cams = tgt_cams[chunk_frame_indices]
                    chunk_meshes = tgt_meshes[chunk_frame_indices]
                    tex = make_repeated_uv_texture(
                        texture,
                        faces_uvs,
                        verts_uvs,
                        N=len(chunk_cams),
                        sampling_mode="nearest",
                    )

                    renderer = make_mesh_renderer(cameras=chunk_cams, resolution=16)
                    render_meshes = chunk_meshes.clone()
                    render_meshes.textures = tex
                    return renderer(render_meshes)

                chunk_rendered_cond = dict_map(textures_cond, render_chunk_frames)
                chunk_rendered_uncond = dict_map(textures_uncond, render_chunk_frames)

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
