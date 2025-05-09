import torch
from tqdm import tqdm

from text3d2video.pipelines.base_pipeline import BaseStableDiffusionPipeline


@torch.no_grad()
def ddim_invert(
    pipe: BaseStableDiffusionPipeline,
    start_latents,
    prompt,
    guidance_scale=3.5,
    num_inference_steps=80,
):
    n_ims = len(start_latents)
    cond_embeddings, uncond_embeddings = pipe.encode_prompt([prompt] * n_ims)

    latents = start_latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps)

    # reversed timesteps
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(range(1, num_inference_steps), total=num_inference_steps - 1):
        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        model_input = torch.cat([latents] * 2)
        both_embeddings = torch.cat([uncond_embeddings, cond_embeddings])
        noise_pred = pipe.unet(
            model_input, t, encoder_hidden_states=both_embeddings
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # Inverted update step (re-arranging the update step to get x(t) (new latents) as a function of x(t-1) (current latents)
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (
            alpha_t_next.sqrt() / alpha_t.sqrt()
        ) + (1 - alpha_t_next).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.stack(intermediate_latents)
