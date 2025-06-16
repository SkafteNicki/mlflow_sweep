from pathlib import Path
import yaml
from mlflow_sweep.models import SweepConfig
from mlflow_sweep.sampler import SweepProcessor
import mlflow
from mlflow.entities import Run
import os
import subprocess
from rich import print as rprint


def init_command(config_path: Path) -> None:
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
    rprint(f"[bold green]Sweep initialized with ID: {run.info.run_id}[/bold green]")


def start_command(sweep_id: str = "") -> None:
    """Start a sweep agent."""
    sweeps: list[Run] = mlflow.search_runs(  # ty: ignore[invalid-assignment]
        search_all_experiments=True,
        filter_string="tag.sweep = 'True'",
        output_format="list",
    )

    if sweep_id:
        for sweep in sweeps:
            if sweep.info.run_id == sweep_id:
                break
        else:
            raise ValueError(f"No sweep found with sweep_id: {sweep_id}")
    else:
        sweep = max(sweeps, key=lambda x: x.info.start_time)  # Get the most recent sweep

    sweep_config_path = [
        a.path for a in mlflow.artifacts.list_artifacts(run_id=sweep.info.run_id) if a.path.endswith(".yaml")
    ][0]
    artifact_uri = sweep.info.artifact_uri.replace("file://", "")  # Remove the 'file://' prefix
    config_file_path = os.path.join(artifact_uri, sweep_config_path)

    with open(config_file_path, "r") as file:
        config = yaml.safe_load(file)

    config = SweepConfig(**config)
    sweep_processor = SweepProcessor(config, parent_sweep=sweep)

    mlflow.set_experiment(experiment_id=sweep.info.experiment_id)
    with mlflow.start_run(run_id=sweep.info.run_id):
        # Set an environment variable to link runs in the sweep
        # This will be picked up by the custom SweepRunContextProvider
        env = os.environ.copy()
        env["SWEEP_PARENT_RUN_ID"] = sweep.info.run_id

        while True:
            output = sweep_processor.propose_next()
            if output is None:
                rprint("[bold red]No more runs can be proposed or run cap reached.[/bold red]")
                break
            command, data = output
            table_data = {k: [str(v)] for k, v in data.items()}
            mlflow.log_table(
                data=table_data,
                artifact_file="proposed_parameters.json",
            )
            rprint(f"[bold blue]Executed command:[/bold blue] \n[italic]{command}[/italic]")
            rprint(50 * "─")
            subprocess.run(command, shell=True, env=env, check=True)
            rprint(50 * "─")


def finalize_command(sweep_id: str = "") -> None:
    """Finalize a sweep."""
    print(sweep_id)
