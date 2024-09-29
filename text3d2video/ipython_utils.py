from typing import List

import matplotlib.pyplot as plt
from PIL.Image import Image


def display_ims(images: List[Image], scale=1):

    if len(images) == 1:
        _, ax = plt.subplots(1, 1, figsize=(scale, scale))
        ax.imshow(images[0])
        ax.axis("off")
        plt.show()
        plt.tight_layout()
        plt.show()
        return

    _, axs = plt.subplots(1, len(images), figsize=(len(images) * scale, scale))

    for i, im in enumerate(images):
        axs[i].imshow(im)
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()
