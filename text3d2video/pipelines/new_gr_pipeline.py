from typing import List, Optional

import numpy as np
import torch
from attr import dataclass
from PIL import Image
from pytorch3d.renderer.mesh.rasterizer import Fragments
from torch import Tensor
from tqdm import tqdm

from text3d2video.backprojection import (
    TexelProjection,
    aggregate_views_uv_texture,
    compute_texel_projection,
)
from text3d2video.mip import view_mip_level, view_uv_res
from text3d2video.pipelines.base_pipeline import (
    PipelineOutput,
)
from text3d2video.pipelines.controlnet_pipeline import BaseControlNetPipeline
from text3d2video.pipelines.generative_rendering_pipeline import (
    GenerativeRenderingConfig,
    GrExtractedFeatures,
    GrLogic,
)
from text3d2video.rendering import (
    AnimSequence,
    make_mesh_rasterizer,
    render_depth_map,
    render_texture,
    shade_texture,
)
from text3d2video.util import cluster_by_threshold, dict_map
from text3d2video.utilities.logging import NULL_LOGGER, setup_greenlists


@dataclass
class GrSrcFrags:
    frags: List[Fragments]
    depth: Image.Image
    projections: List[TexelProjection]  # projection at each resolution


@dataclass
class GrTgtFrags:
    frags: Fragments
    depth: Image.Image
    cluster_idx: int


@dataclass
class MipGrFrags:
    src_frags: List[GrSrcFrags]
    tgt_frags: List[GrTgtFrags]
    src_clusters: List[List[int]]
    uv_resolutions: List[List[int]]


def precompute_frags(
    src_seq: AnimSequence,
    tgt_seq: AnimSequence,
    render_resolutions: List[int],
    with_multires=True,
) -> MipGrFrags:
    # compute mip level of source frames
    src_mip_levels = [
        view_mip_level(c, m, src_seq.verts_uvs, src_seq.faces_uvs)
        for c, m in zip(src_seq.cams, src_seq.meshes)
    ]

    # cluster source frames by mip level
    src_clusters = cluster_by_threshold(src_mip_levels, threshold=0.05)

    if not with_multires:
        src_clusters = cluster_by_threshold(src_mip_levels, threshold=100)

    # single frame for each cluster for computign uv resolutions
    cluster_representatives = [
        cluster[0] for cluster in src_clusters for _ in range(len(cluster))
    ]
    cluster_mip_levels = np.array([src_mip_levels[i] for i in cluster_representatives])

    # compute UV resolution for each cluster representative
    uv_resolutions = []  # (frames, render_resolutions)
    for i in cluster_representatives:
        resolutions = []
        for res in render_resolutions:
            uv_resolution = view_uv_res(
                src_seq.cams[i],
                src_seq.meshes[i],
                src_seq.verts_uvs,
                src_seq.faces_uvs,
                res,
            )
            resolutions.append(uv_resolution)
        uv_resolutions.append(resolutions)

    # for each source frame, compute:
    # 1. rasterization frags at each render resolution
    # 2. texel projection at each uv resolution
    # 3. depth map
    src_frags = []
    for i in range(len(src_seq)):
        mesh = src_seq.meshes[i]
        cam = src_seq.cams[i]
        verts_uvs = src_seq.verts_uvs
        faces_uvs = src_seq.faces_uvs

        # compute rasterization frags to render resolutions
        fragments = []
        for res in render_resolutions:
            rasterizer = make_mesh_rasterizer(resolution=res)
            f = rasterizer(mesh, cameras=cam)
            fragments.append(f)

        # compute projections to all uv_resolutions
        frame_resolutions = uv_resolutions[i]
        projections = []
        for uv_res in frame_resolutions:
            p = compute_texel_projection(mesh, cam, verts_uvs, faces_uvs, uv_res)
            projections.append(p)

        # render depth map
        depth = render_depth_map(mesh, cam)[0]
        src_frags.append(GrSrcFrags(fragments, depth, projections))

    # for each target frame compute:
    # 1. rasterization frags at each render resolution
    # 2. depth map
    # 3. index of cluster frame with closest mip level
    tgt_frags = []
    for i in range(len(tgt_seq)):
        mesh = tgt_seq.meshes[i]
        cam = tgt_seq.cams[i]
        verts_uvs = tgt_seq.verts_uvs
        faces_uvs = tgt_seq.faces_uvs

        # get index of source frame with closest mip level
        mip_level = view_mip_level(cam, mesh, verts_uvs, faces_uvs)
        src_index = np.argmin(np.abs(cluster_mip_levels - mip_level))

        # compute rasterization frags to render resolutions
        fragments = []
        for res in render_resolutions:
            rasterizer = make_mesh_rasterizer(resolution=res)
            f = rasterizer(mesh, cameras=cam)
            fragments.append(f)

        # render depth map
        depth = render_depth_map(mesh, cam)[0]
        tgt_frags.append(
            GrTgtFrags(
                frags=fragments,
                depth=depth,
                cluster_idx=src_index,
            )
        )

    print(src_clusters)
    print([f.cluster_idx for f in tgt_frags])

    return MipGrFrags(
        src_frags=src_frags,
        tgt_frags=tgt_frags,
        src_clusters=src_clusters,
        uv_resolutions=uv_resolutions,
    )


class GrPipelineNew(BaseControlNetPipeline):
    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        tgt_seq: AnimSequence,
        src_seq: AnimSequence,
        src_latents: Tensor,
        conf: GenerativeRenderingConfig,
        multi_res_textures=True,
        initial_texture: Optional[Tensor] = None,
        texture_noise_level: Optional[float] = None,
        generator: Optional[torch.Generator] = None,
        logger=NULL_LOGGER,
    ):
        # setup GR logic
        gr = GrLogic.from_gr_config(self, conf)
        gr.set_attn_processor()

        # precompute projections and fragments for all resolutions/frames
        render_resolutions, layer_resolution_indices = gr.calculate_layer_resolutions()
        mip_frags = precompute_frags(
            src_seq, tgt_seq, render_resolutions, with_multires=multi_res_textures
        )

        n_frames = len(tgt_seq)

        # encode source prompt
        src_cond_embs, src_uncond_embs = self.encode_prompt(
            [prompt] * len(src_seq.cams)
        )

        # Get prompt embeddings
        cond_embeddings, uncond_embeddings = self.encode_prompt([prompt] * n_frames)

        # initialize noise
        latents = self.prepare_latents(len(tgt_seq), generator=generator)

        self.scheduler.set_timesteps(conf.num_inference_steps)
        timesteps = self.scheduler.timesteps

        do_texture_initialization = initial_texture is not None
        if do_texture_initialization:
            assert (
                texture_noise_level is not None
            ), "texture_noise_level must be provided"

            timesteps = self.get_partial_timesteps(
                conf.num_inference_steps, texture_noise_level
            )

            # TODO when you get features get it from shifted layer
            start_t = timesteps[0]
            renders = render_texture(
                tgt_seq.meshes,
                tgt_seq.cams,
                initial_texture,
                tgt_seq.verts_uvs,
                tgt_seq.faces_uvs,
            )
            noise = latents
            encoded = self.encode_images(renders)
            latents = self.scheduler.add_noise(encoded, noise, start_t)

        setup_greenlists(logger, timesteps, n_frames, n_save_frames=10, n_save_times=10)

        # chunk indices to use i regeneration
        chunks_indices = torch.split(torch.arange(0, n_frames), conf.chunk_size)

        all_latents = {}

        # denoising loop
        for i, t in enumerate(tqdm(timesteps)):
            all_latents[t.item()] = latents

            # extract features from source frames
            extr_latents = src_latents[t.item()]
            extr_depth_maps = [f.depth for f in mip_frags.src_frags]
            feats = gr.model_forward_extraction(
                extr_latents,
                src_cond_embs,
                src_uncond_embs,
                extr_depth_maps,
                t,
            )

            def project_feats(layer, feats):
                return project_feats_multires(
                    layer, feats, mip_frags, layer_resolution_indices
                )

            cond_textures = dict_map(feats.cond_post_attn, project_feats)
            uncond_textures = dict_map(feats.uncond_post_attn, project_feats)

            # denoise in chunks with feature injection
            noise_preds = []
            for chunk_frame_indices in chunks_indices:

                def render_chunk(layer, textures):
                    return render_multires_texture(
                        layer,
                        textures,
                        mip_frags,
                        layer_resolution_indices,
                        chunk_frame_indices.tolist(),
                        tgt_seq,
                    )

                renders_cond = dict_map(cond_textures, render_chunk)
                renders_uncond = dict_map(uncond_textures, render_chunk)

                depth_maps = [
                    mip_frags.tgt_frags[i].depth for i in chunk_frame_indices.tolist()
                ]

                # model forward with pre-and post-attn injection
                noise_pred = gr.model_forward_injection(
                    latents[chunk_frame_indices],
                    cond_embeddings[chunk_frame_indices],
                    uncond_embeddings[chunk_frame_indices],
                    depth_maps,
                    t,
                    feats.cond_kv,
                    feats.uncond_kv,
                    renders_cond,
                    renders_uncond,
                )

                noise_preds.append(noise_pred)

            noise_preds = torch.cat(noise_preds, dim=0)
            latents = self.scheduler.step(
                noise_preds, t, latents, generator=generator
            ).prev_sample

        all_latents[0] = latents

        ims = self.decode_latents(
            latents, chunk_size=conf.chunk_size, generator=generator
        )

        return PipelineOutput(images=ims, latents=all_latents)


def project_to_textures(
    feats: GrExtractedFeatures, mip_frags: MipGrFrags, layer_resolution_indices, gr
):
    cluster_cond_textures = []
    cluster_uncond_textures = []
    for cluster in mip_frags.src_clusters:
        # get feats for the cluster
        def get_cluster_features(layer, feats):
            return feats[cluster]

        clust_features = dict_map(feats.cond_post_attn, get_cluster_features)
        clust_feats_uncond = dict_map(feats.uncond_post_attn, get_cluster_features)

        # obtain texture for each cluster
        def aggr_feats(layer, features):
            rep_i = cluster[0]
            res_idx = layer_resolution_indices[layer]
            uv_res = mip_frags.src_frags[rep_i].projections[res_idx].uv_resolution
            projections = [mip_frags.src_frags[i].projections[res_idx] for i in cluster]
            return gr.aggregate_features(features, uv_res, projections)

        tex_cond = dict_map(clust_features, aggr_feats)
        tex_uncond = dict_map(clust_feats_uncond, aggr_feats)

        cluster_cond_textures.append(tex_cond)
        cluster_uncond_textures.append(tex_uncond)

    return cluster_cond_textures, cluster_uncond_textures


def project_feats_multires(layer, feats, mip_frags, layer_resolution_indices):
    textures = []
    for cluster in mip_frags.src_clusters:
        cluster_feats = feats[cluster]

        rep = cluster[0]
        res_idx = layer_resolution_indices[layer]
        uv_res = mip_frags.src_frags[rep].projections[res_idx].uv_resolution
        projections = [mip_frags.src_frags[i].projections[res_idx] for i in cluster]

        texture = aggregate_views_uv_texture(cluster_feats, uv_res, projections)
        textures.append(texture)

    return textures


def render_multires_texture(
    layer: str,
    multires_texture: List[torch.Tensor],
    mip_frags: MipGrFrags,
    layer_resolution_indices: List[int],
    frame_indices: List[int],
    tgt_seq: AnimSequence,
):
    res_idx = layer_resolution_indices[layer]
    renders = []
    for i in frame_indices:
        mesh = tgt_seq.meshes[i]
        cluster_idx = mip_frags.tgt_frags[i].cluster_idx
        tex = multires_texture[cluster_idx]

        frags = [mip_frags.tgt_frags[i].frags[res_idx]]
        render = shade_texture(
            mesh,
            frags,
            tex,
            tgt_seq.verts_uvs,
            tgt_seq.faces_uvs,
            sampling_mode="nearest",
        )[0]
        renders.append(render)

    renders = torch.stack(renders)
    return renders
