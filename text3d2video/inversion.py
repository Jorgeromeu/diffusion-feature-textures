import torch
from tqdm import tqdm


@torch.no_grad()
def ddim_inversion(
    pipe,
    latents,
    prompt,
    guidance_scale=3.5,
    num_inference_steps=80,
    device="cuda",
):
    # encode prompt
    cond_embs, uncond_embs = pipe.encode_prompt(
        [prompt],
    )
    both_embs = torch.cat([uncond_embs, cond_embs])

    latents = latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = []

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(list(range(1, num_inference_steps))):
        # We'll skip the final iteration
        if i >= num_inference_steps - 1:
            continue

        t = timesteps[i]

        latents_both = torch.cat([latents] * 2)
        latents_both = pipe.scheduler.scale_model_input(latents_both, t)

        # model forward
        noise_pred = pipe.unet(latents_both, t, encoder_hidden_states=both_embs).sample

        # classifier-free-guidance
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # inverted update step
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (
            alpha_t_next.sqrt() / alpha_t.sqrt()
        ) + (1 - alpha_t_next).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.cat(intermediate_latents), timesteps


@torch.no_grad()
def my_inversion(
    pipe,
    latents,
    prompt,
    guidance_scale=3.5,
    num_inference_steps=80,
    device="cuda",
):
    # encode prompt
    cond_embs, uncond_embs = pipe.encode_prompt(
        [prompt] * len(latents),
    )
    both_embs = torch.cat([uncond_embs, cond_embs])

    latents = latents.clone()

    # We'll keep a list of the inverted latents as the process goes on
    intermediate_latents = [latents]

    # Set num inference steps
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = reversed(pipe.scheduler.timesteps)

    for i in tqdm(list(range(1, num_inference_steps))):
        # We'll skip the final iteration
        # if i >= num_inference_steps - 1:
        #     continue

        t = timesteps[i]

        latents_both = torch.cat([latents] * 2)
        latents_both = pipe.scheduler.scale_model_input(latents_both, t)

        # model forward
        noise_pred = pipe.unet(latents_both, t, encoder_hidden_states=both_embs).sample

        # classifier-free-guidance
        noise_pred_cond, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

        current_t = max(0, t.item() - (1000 // num_inference_steps))  # t
        next_t = t  # min(999, t.item() + (1000//num_inference_steps)) # t+1
        alpha_t = pipe.scheduler.alphas_cumprod[current_t]
        alpha_t_next = pipe.scheduler.alphas_cumprod[next_t]

        # inverted update step
        latents = (latents - (1 - alpha_t).sqrt() * noise_pred) * (
            alpha_t_next.sqrt() / alpha_t.sqrt()
        ) + (1 - alpha_t_next).sqrt() * noise_pred

        # Store
        intermediate_latents.append(latents)

    return torch.stack(intermediate_latents), timesteps
