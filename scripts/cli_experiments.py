import typer

from scripts.wandb_experiments.benchmark import benchmark

app = typer.Typer()

experiment_funs = [
    benchmark,
]

experiment_funs = {f.__name__: f for f in experiment_funs}

# have a way of specifying with a yaml file
# the experiment config, and experiment name


@app.command()
def main(config_path: str):
    pass
