class DiffusionLogger:
    pass


class Writer:
    def __init__(self, logger: DiffusionLogger):
        self.logger = logger


# Implementation


class LatentsWriter(Writer):
    def write_noisy_latent(self, latent, t):
        pass


class TexGenLogger:
    latent_writer: LatentsWriter

    def __init__(self):
        super().__init__()
        self.latent_writer = LatentsWriter(self)
