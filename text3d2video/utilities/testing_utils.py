import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont


def process_color(color):
    if isinstance(color, str):
        return color

    if isinstance(color, torch.Tensor):
        if color.shape != (3,):
            raise ValueError("Color tensor must have shape (3,)")
        r = color[0].item()
        g = color[1].item()
        b = color[2].item()

    if isinstance(color, tuple):
        if len(color) != 3:
            raise ValueError("Color tuple must have length 3")
        r, g, b = color

    if isinstance(color, np.ndarray):
        if color.shape != (3,):
            raise ValueError("Color array must have shape (3,)")
        r, g, b = color

    if isinstance(color, list):
        if len(color) != 3:
            raise ValueError("Color list must have length 3")
        r, g, b = color

    rgb = [r, g, b]

    if all(isinstance(c, int) for c in rgb):
        return (r, g, b)

    if all(isinstance(c, float) for c in rgb):
        return tuple(int(c * 255) for c in color)

    raise ValueError("Invalid color format")


def test_img(
    txt: str = "",
    resolution=200,
    font_percent=0.3,
    color="lightgray",
    textcolor="white",
    font="arial.ttf",
    return_type="PIL",
):
    color = process_color(color)
    textcolor = process_color(textcolor)

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


def checkerboard_img(
    res=100,
    square_size=10,
    color1=(250, 250, 250),
    color2=(200, 200, 200),
    return_type="PIL",
):
    color1 = process_color(color1)
    color2 = process_color(color2)

    size = (res, res)

    width, height = size
    img = Image.new("RGB", size)
    pixels = img.load()

    for y in range(height):
        for x in range(width):
            pass
            # Determine if the square should be color1 or color2
            if ((x // square_size) + (y // square_size)) % 2 == 0:
                pixels[x, y] = color1
            else:
                pixels[x, y] = color2

    if return_type == "PIL":
        return img

    # Convert the image to a PyTorch tensor
    return TF.to_tensor(img)


def gradient_img(res=100, color_start=(0, 0, 255), color_end=(255, 0, 0)):
    # Define the image size and colors for the gradient
    width, height = (res, res)

    # Create an array to hold the gradient data
    gradient = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the gradient by interpolating from the start color to the end color
    for y in range(height):
        for x in range(width):
            # Determine the interpolation factor based on distance along the diagonal
            factor = (x + y) / (width + height - 2)
            r = int(color_start[0] * (1 - factor) + color_end[0] * factor)
            g = int(color_start[1] * (1 - factor) + color_end[1] * factor)
            b = int(color_start[2] * (1 - factor) + color_end[2] * factor)

            # Assign the calculated color to each pixel
            gradient[y, x] = (r, g, b)

    # Convert the numpy array to an image and save/show it
    image = Image.fromarray(gradient)
    return image
