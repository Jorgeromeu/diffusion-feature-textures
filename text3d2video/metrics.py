import torch
from omegaconf import OmegaConf
from transformers import CLIPModel, CLIPProcessor

from text3d2video.artifacts.video_artifact import VideoArtifact


class CLIPMetrics:
    def __init__(self, model_repo="openai/clip-vit-base-patch32"):
        self.device = torch.device("cuda")
        self.torch_dtype = torch.float16

        self.model = CLIPModel.from_pretrained(
            model_repo,
            device_map=self.device,
            torch_dtype=self.torch_dtype,
        )
        self.processor = CLIPProcessor.from_pretrained(model_repo)

    def get_clip_output(self, video: VideoArtifact):
        # get prompt
        generation_run = video.logged_by()
        prompt = OmegaConf.create(generation_run.config).prompt

        # get frames
        frames = video.get_frames()

        inputs = self.processor(text=[prompt], images=frames, return_tensors="pt")
        inputs.to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs

    def prompt_fidelity(self, video: VideoArtifact):
        outs = self.get_clip_output(video)

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

    def frame_consistency(self, video: VideoArtifact):
        outs = self.get_clip_output(video)
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
