# %%
import matplotlib.pyplot as plt
import matplotlib
from moviepy.editor import VideoFileClip
from pathlib import Path
import numpy as np

scene = 'house'
index = 1
data_path = Path('../data/generative_rendering_results_processed')
figures_path = Path('../outs/figures')
video_path = data_path / scene / f'out_{index}.mp4'
uv_path = data_path / scene / f'uv.mp4'
# %%
video = VideoFileClip(str(video_path))
uv = VideoFileClip(str(uv_path))

timesteps = [1.9, 2.067, 2.50, 2.93]
N= len(timesteps)

frames = [video.get_frame(t) for t in timesteps]

scale = 2.5
fig, axs = plt.subplots(1, N, figsize=(N*scale, 1*scale))

for i, t in enumerate(timesteps):

    frame = video.get_frame(t)

    axs[i].imshow(frame)
    axs[i].axis('off')
    axs[i].set_title(f't={t:.2f}')

fig.tight_layout()
plt.savefig(figures_path / 'house_long_term.pdf')
plt.show()