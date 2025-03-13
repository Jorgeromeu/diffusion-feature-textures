from dataclasses import dataclass

from diffusers.schedulers.scheduling_utils import SchedulerMixin

from text3d2video.utilities.diffusion_data import (
    AttnFeaturesWriter,
    DiffusionDataLogger,
    LatentsWriter,
)
from wandb_util.wandb_util import ArtifactWrapper


@dataclass
class SdDataConfig:
    enabled: bool
    n_frames: int
    n_timesteps: int
    out_artifact: str
    module_paths: list[str]
    save_latents: bool
    save_q: bool
    save_k: bool
    save_v: bool


class SdDataArtifact(ArtifactWrapper):
    wandb_artifact_type = "sd_data"

    # config for saving
    config: SdDataConfig
    diffusion_data: DiffusionDataLogger
    # diffusion data writers
    attn_writer: AttnFeaturesWriter
    latents_writer: LatentsWriter

    def h5_file_path(self):
        return self.folder / "data.h5"

    @classmethod
    def init_from_config(cls, config: SdDataConfig):
        art = SdDataArtifact.create_empty_artifact(config.out_artifact)
        art.config = config

        # diffusion data
        art.diffusion_data = DiffusionDataLogger(
            art.h5_file_path(),
            enabled=config.enabled,
            path_greenlist=config.module_paths,
        )

        # writers
        art.latents_writer = LatentsWriter(
            art.diffusion_data, enabled=config.save_latents
        )
        art.attn_writer = AttnFeaturesWriter(
            art.diffusion_data,
            save_q=config.save_q,
            save_k=config.save_k,
            save_v=config.save_v,
        )
        return art

    def begin_recording(self, scheduler: SchedulerMixin, n_frames: int):
        self.diffusion_data.calculate_evenly_spaced_save_noise_levels(scheduler)
        self.diffusion_data.calc_evenly_spaced_frame_indices(n_frames)
        self.diffusion_data.begin_recording()

    def end_recording(self):
        self.diffusion_data.end_recording()

    # Reading methods
