from pathlib import Path
import yaml
from mlflow_sweep.models import SweepConfig
import mlflow


def init_command(config_path: Path):
    """Start a sweep from a config.

    Args:
        config_path (Path): Path to the sweep configuration file.

    """
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    config = SweepConfig(**config)  # validate the config

    mlflow.set_experiment(config.experiment_name)
    run = mlflow.start_run(run_name=config.sweep_name)
    mlflow.set_tag("sweep", True)
    mlflow.log_artifact(str(config_path))
    print(run.info.run_id)


def start_command(sweep_id: str = ""):
    """Start a sweep agent."""
    pass


def finalize_command(sweep_id: str = ""):
    """Finalize a sweep."""
    pass
