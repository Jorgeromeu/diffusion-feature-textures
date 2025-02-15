import torch.nn as nn
import torchvision.transforms.functional as TF


def affine(im, translation=(0, 0), scale=1.0, angle=0.0):
    h, w = TF.get_image_size(im)

    return TF.affine(
        im,
        translate=(translation[0] * h, translation[1] * w),
        scale=scale,
        angle=angle,
        shear=0.0,
    )


def affine_inv(im, translation=(0, 0), scale=1.0, angle=0.0):
    h, w = TF.get_image_size(im)
    return TF.affine(
        im,
        translate=(-translation[0] * h, -translation[1] * w),
        scale=1 / scale,
        angle=-angle,
        shear=0.0,
    )


class Affine2D(nn.Module):
    def __init__(self, translate=(0, 0), scale=1.0, angle=0.0):
        super().__init__()
        self.translate = translate
        self.scale = scale
        self.angle = angle

    def forward(self, x):
        h, w = TF.get_image_size(x)
        return TF.affine(
            x,
            translate=(self.translate[0] * h, self.translate[1] * w),
            scale=self.scale,
            angle=self.angle,
            shear=0.0,
        )

    def inverse(self):
        return Affine2D(
            translate=(-self.translate[0], -self.translate[1]),
            scale=1 / self.scale,
            angle=-self.angle,
        )
