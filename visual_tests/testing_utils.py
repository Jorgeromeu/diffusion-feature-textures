import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont


def random_color():
    color_float = torch.randn(3)
    color_int = (color_float * 255).int()

    return tuple(color_int.tolist())


def test_img(
    txt: str = "",
    resolution=200,
    font_percent=0.3,
    color="lightgray",
    textcolor="white",
    font="arial.ttf",
    return_type="PIL",
):
    image_size = (resolution, resolution)
    gray_image = Image.new(
        "RGB", image_size, color=color
    )  # RGB mode with mid-gray color

    # Draw the number on the image
    draw = ImageDraw.Draw(gray_image)
    font_size = resolution * font_percent
    font = ImageFont.truetype("arial.ttf", font_size)

    # Get bounding box of the text
    bbox = draw.textbbox((0, 0), txt, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    text_position = (
        (image_size[0] - text_width) / 2,
        (image_size[1] - text_height) / 2,
    )

    draw.text(text_position, txt, fill=textcolor, font=font)

    if return_type == "PIL":
        return gray_image

    # Convert the image to a PyTorch tensor
    return TF.to_tensor(gray_image)
