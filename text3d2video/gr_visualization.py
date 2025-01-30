import shutil
from pathlib import Path
from typing import List

from diffusers import UNet2DConditionModel
from einops import rearrange

from text3d2video.artifacts.gr_data import GrDataArtifact
from text3d2video.feature_visualization import RgbPcaUtil, reduce_feature_map
from text3d2video.sd_feature_extraction import AttnLayerId
from text3d2video.utilities.attention_visualization import reshape_concatenated


def write_gr_images(
    art: GrDataArtifact,
    path: Path,
    time: int,
    layers: List[AttnLayerId],
    frame_indices: List[int],
    unet: UNet2DConditionModel,
):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(exist_ok=True)

    for layer in layers:
        module = layer.module_path()
        module_folder = path / module
        module_folder.mkdir()

        layer_res = layer.layer_resolution(unet)

        rendered_frames = [
            art.gr_writer.read_post_attn_post_injection(time, f, module)
            for f in frame_indices
        ]

        keyframe_post_attn = art.gr_writer.read_kf_post_attn(time, module)[0]

        key = art.attn_writer.read_key(time, 0, module)
        val = art.attn_writer.read_val(time, 0, module)
        key_square = reshape_concatenated(key, layer_res)
        val_square = reshape_concatenated(val, layer_res)

        # compute k/v pca
        key_rgb = reduce_feature_map(key_square)
        val_rgb = reduce_feature_map(val_square)

        key_rgb.save(module_folder / "key.png")
        val_rgb.save(module_folder / "val.png")

        # compute post atn PCA
        features = rearrange(keyframe_post_attn, "b c h w -> (b h w) c")
        pca = RgbPcaUtil.init_from_features(features)

        rendered_frames_rgb = [pca.feature_map_to_rgb_pil(f) for f in rendered_frames]

        kf_post_attn_frames_rgb = [
            pca.feature_map_to_rgb_pil(f) for f in keyframe_post_attn
        ]

        keyframes_folder = module_folder / "keyframes"
        rendered_folder = module_folder / "rendered"
        keyframes_folder.mkdir()
        rendered_folder.mkdir()

        for i, frame in enumerate(kf_post_attn_frames_rgb):
            frame.save(keyframes_folder / f"frame_{i}.png")

        for i, frame in enumerate(rendered_frames_rgb):
            frame.save(rendered_folder / f"frame_{i}.png")
