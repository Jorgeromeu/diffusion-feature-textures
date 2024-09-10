# %%
from diffusers import StableDiffusionPipeline, DDIMScheduler
import requests
from PIL import Image
from io import BytesIO

# %%
sd_repo = 'runwayml/stable-diffusion-v1-5'
device = 'cuda'

pipe = StableDiffusionPipeline.from_pretrained(sd_repo).to(device)

# %%
pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

# %%
out = pipe('Deadpool and Wolverine', num_inference_steps=25)
out.images[0].resize((250, 250))
# %%

timesteps = pipe.scheduler.timesteps.cpu()
alphas = pipe.scheduler.alphas_cumprod[timesteps]

# %%
def load_image(url, size=None):
    response = requests.get(url,timeout=0.2)
    img = Image.open(BytesIO(response.content)).convert('RGB')
    if size is not None:
        img = img.resize(size)
    return img

input_image = load_image('https://images.pexels.com/photos/8306128/pexels-photo-8306128.jpeg', size=(512, 512))
input_image