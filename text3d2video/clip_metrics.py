from typing import List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from transformers.models.clip.modeling_clip import CLIPOutput


class CLIPMetrics:
    """
    Utility class for evaluating quality of output videos
    """

    def __init__(self, model_repo="openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda")
        self.torch_dtype = torch.float16

        self.model = CLIPModel.from_pretrained(
            model_repo,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
        )
        self.processor = CLIPProcessor.from_pretrained(model_repo)

    def model_forward(self, images: List[Image.Image], texts: List[str]) -> CLIPOutput:
        inputs = self.processor(text=texts, images=images, return_tensors="pt")
        inputs.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs

    def prompt_fidelity(self, frames: List[Image.Image], prompt: str):
        outs = self.model_forward(frames, [prompt])

        image_embeds = outs.image_embeds
        text_embed = outs.text_embeds[0]

        similarities = []
        for embedding in image_embeds:
            sim = torch.cosine_similarity(
                embedding.unsqueeze(0), text_embed.unsqueeze(0)
            )
            similarities.append(sim.item())

        average_sim = torch.mean(torch.tensor(similarities)).item()
        return average_sim

    def frame_consistency(self, frames: List[Image.Image]):
        outs = self.model_forward(frames, [""])
        image_embeds = outs.image_embeds

        similarities = []
        for i in range(len(image_embeds) - 1):
            embedding = image_embeds[i]
            next_embedding = image_embeds[i + 1]

            sim = torch.cosine_similarity(
                embedding.unsqueeze(0), next_embedding.unsqueeze(0)
            )
            similarities.append(sim.item())

        average_sim = torch.mean(torch.tensor(similarities)).item()
        return average_sim
