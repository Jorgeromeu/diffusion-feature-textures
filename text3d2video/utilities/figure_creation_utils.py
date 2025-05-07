from pathlib import Path


def write_image_seq(folder: Path, name, images):
    seq_path = folder / name
    seq_path.mkdir(parents=True, exist_ok=True)
    for i, im in enumerate(images):
        im.save(seq_path / f"im_{i:03d}.png")
