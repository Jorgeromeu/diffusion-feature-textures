from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from einops import rearrange

from text3d2video.feature_visualization import RgbPcaUtil
from text3d2video.pipelines.generative_rendering_pipeline import (
    GrOutput,
)
from text3d2video.util import create_fresh_dir, hwc_to_chw, pil_latent
from text3d2video.utilities.attn_vis import reshape_concatenated
from text3d2video.utilities.figure_creation_utils import write_image_seq
from text3d2video.utilities.logging import H5Logger


def main_pipeline_figure_images(out_path: str, h5_path: str, out: GrOutput) -> None:
    """
    Read
    """

    path = create_fresh_dir(out_path)

    logger = H5Logger(Path(h5_path))
    logger.open_read()

    to_pil = TF.to_pil_image

    fig_frame_is = [0, 4, 8, 19]

    # Read anim renders, encoded and initial latents
    anim_renders = [
        logger.read("anim_render", frame_i=f, transform=to_pil) for f in fig_frame_is
    ]

    anim_encoded = [
        logger.read("anim_encoded", frame_i=f, transform=pil_latent)
        for f in fig_frame_is
    ]

    anim_latents = [
        logger.read("anim_latent", frame_i=f, transform=pil_latent)
        for f in fig_frame_is
    ]

    # write anim renders/encoded/initial noise
    write_image_seq(path, "anim_renders", anim_renders)
    write_image_seq(path, "anim_encoded", anim_encoded)
    write_image_seq(path, "anim_latents", anim_latents)

    extr_frames = logger.key_values("extr_render", "extr_frame_i")

    extr_renders = [
        logger.read("extr_render", extr_frame_i=f, transform=to_pil)
        for f in extr_frames
    ]

    extr_encoded = [
        logger.read("extr_encoded", extr_frame_i=f, transform=pil_latent)
        for f in extr_frames
    ]

    extr_latents = [
        logger.read("extr_latent", extr_frame_i=f, transform=pil_latent)
        for f in extr_frames
    ]

    # write extraction renders/encoded/initial noise
    write_image_seq(path, "extr_renders", extr_renders)
    write_image_seq(path, "extr_encoded", extr_encoded)
    write_image_seq(path, "extr_latents", extr_latents)

    # write outputs
    write_image_seq(path, "anim_output", [out.images[i] for i in fig_frame_is])
    write_image_seq(path, "extr_output", out.extr_images)

    ts = sorted(logger.key_values("feats_cond", "t"))
    layers = sorted(logger.key_values("feats_cond", "layer"))

    layer = layers[-1]
    t = ts[len(ts) // 2]
    tgt_frame = fig_frame_is[len(fig_frame_is) // 2]

    write_feature_ims(path, logger, t, layer, tgt_frame)


def write_feature_ims(
    path: Path, logger: H5Logger, t: int, layer: str, tgt_frame: 0
) -> None:
    # Read extracted feats
    extr_frame_is = logger.key_values("feats_cond", "extr_frame_i")
    extr_feats = torch.stack(
        [
            logger.read("feats_cond", t=t, extr_frame_i=i, layer=layer)
            for i in extr_frame_is
        ]
    )

    # Read feature texture
    feat_tex = logger.read("feat_tex_cond", t=t, layer=layer)

    # Read Rendered Feat
    render = logger.read("renders_cond", t=t, frame_i=tgt_frame, layer=layer)

    # read kvs
    kvs = logger.read("kvs_cond", t=t, layer=layer)
    kvs_square = reshape_concatenated(kvs, layer_res=render.shape[-1])

    # Compute KVS PCA
    pca_kvs = RgbPcaUtil.init_from_features(kvs)
    kvs_pil = pca_kvs.feature_map_to_rgb_pil(kvs_square)
    kvs_pil.save(path / "kvs.png")

    # compute Features PCA
    feat_tex_flat = rearrange(feat_tex, "h w c -> (h w) c")
    pca = RgbPcaUtil.init_from_features(feat_tex_flat)

    # reduce extracted, texture and render
    feat_tex_pil = pca.feature_map_to_rgb_pil(hwc_to_chw(feat_tex))
    render_pil = pca.feature_map_to_rgb_pil(render)
    extr_feats_pil = [pca.feature_map_to_rgb_pil(feat) for feat in extr_feats]

    write_image_seq(path, "extr_feats", extr_feats_pil)
    render_pil.save(path / "render.png")
    feat_tex_pil.save(path / "feat_tex.png")
