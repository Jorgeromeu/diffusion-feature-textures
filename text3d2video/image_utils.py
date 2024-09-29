from einops import rearrange
import torch
import torchvision.transforms.functional as TF


def to_pil_image(image_tensor, channels_last=False):

    if channels_last:
        image_tensor = move_channels_first(image_tensor)

    image_tensor_float = torch.FloatTensor(image_tensor)
    return TF.to_pil_image(image_tensor_float)


def to_tensor(image, channels_last=False):
    tensor = TF.to_tensor(image)
    if channels_last:
        tensor = move_channels_last(tensor)
    return tensor


def move_channels_first(
    im,
):
    return rearrange(im, "h w c -> c h w")


def move_channels_last(
    im,
):
    return rearrange(im, "c h w -> h w c")
