import typer
from omegaconf import OmegaConf

import wandb_util.wandb_util as wbu

app = typer.Typer()


@app.command()
def run_experiment(config_path: str):
    # read config
    exp = OmegaConf.load(config_path)
    exp = wbu.ExperimentConfig(**exp)
    # sync experiment
    exp.sync_experiment()


if __name__ == "__main__":
    app()
