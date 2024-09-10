# %%
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# Open the GIF file
gif_path = Path("../outs/deadpool.gif")
gif = Image.open(gif_path)

def extract_frames(gif, n):
    frames = []
    for i in range(n):
        try:
            gif.seek(gif.n_frames // n * i)
            frame = gif.copy()
            frames.append(frame)
        except EOFError:
            break
    return frames

frames = extract_frames(gif, 5)

fig, ax = plt.subplots(1, 5, figsize=(20, 4))

for i in range(5):
    ax[i].axis('off')
    ax[i].imshow(frames[i])
plt.tight_layout()