from typing import List

import matplotlib.pyplot as plt
from PIL.Image import Image


def display_ims(images: List[Image], scale=1):

    _, axs = plt.subplots(1, len(images), figsize=(len(images) * scale, scale))

    for i, im in enumerate(images):
        axs[i].imshow(im)
        axs[i].axis("off")

    plt.tight_layout()
    plt.show()
