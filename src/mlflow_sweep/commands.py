import os
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from mlflow.entities import Run
from rich import print as rprint
from rich.console import Console
from rich.table import Table

from mlflow_sweep.models import SweepConfig
from mlflow_sweep.plotting import plot_metric_vs_time, plot_parameter_importance_and_correlation
from mlflow_sweep.sampler import SweepSampler
from mlflow_sweep.sweepstate import SweepState
from mlflow_sweep.utils import calculate_feature_importance_and_correlation, current_time_convert


def determine_sweep(sweep_id: str) -> Run:
    """Determine the sweep to use.
    If a sweep_id is provided, it will be used. Otherwise, the most recent sweep will be selected."""

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

    return sweep


def init_command(config_path: Path) -> None:
    """Start a sweep from a config.

    Args:
        config_path (Path): Path to the sweep configuration file.

    """
    with open(config_path) as file:
        config = yaml.safe_load(file)

    config = SweepConfig(**config)  # validate the config
    rprint("[bold blue]Initializing sweep with configuration:[/bold blue]")
    rprint(config)

    mlflow.set_experiment(config.experiment_name)
    run = mlflow.start_run(run_name=config.sweep_name)
    mlflow.set_tag("sweep", True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_config_path = Path(tmpdir) / "sweep_config.yaml"
        shutil.copy(config_path, tmp_config_path)
        mlflow.log_artifact(str(tmp_config_path))

    rprint(f"[bold green]Sweep initialized with ID: {run.info.run_id}[/bold green]")


def run_command(sweep_id: str = "") -> None:
    """Run a sweep agent."""
    sweep = determine_sweep(sweep_id)

    config = SweepConfig.from_sweep(sweep)
    runstate = SweepState(sweep_id=sweep.info.run_id)
    sweep_sampler = SweepSampler(config, runstate)

    mlflow.set_experiment(experiment_id=sweep.info.experiment_id)
    mlflow.start_run(run_id=sweep.info.run_id)

    # Set an environment variable to link runs in the sweep
    # This will be picked up by the custom SweepRunContextProvider
    global_env = os.environ.copy()
    global_env["SWEEP_PARENT_RUN_ID"] = sweep.info.run_id
    global_env["SWEEP_AGENT_ID"] = str(uuid.uuid4())  # Unique ID for this agent

    while True:
        output = sweep_sampler.propose_next()
        if output is None:
            rprint("[bold red]No more runs can be proposed or run cap reached.[/bold red]")
            break
        command, data = output
        mlflow.log_table(
            data={k: [v] for k, v in data.items()},
            artifact_file="proposed_parameters.json",
        )
        rprint(f"[bold blue]Executed command:[/bold blue] \n[italic]{command}[/italic]")
        rprint(50 * "─")
        local_env = global_env.copy()
        local_env["SWEEP_RUN_ID"] = data["sweep_run_id"]
        subprocess.run(command, shell=True, env=local_env, check=True)
        rprint(50 * "─")


def finalize_command(sweep_id: str = "") -> None:
    """Finalize a sweep."""
    sweep = determine_sweep(sweep_id)
    config = SweepConfig.from_sweep(sweep)
    runstate = SweepState(sweep_id=sweep.info.run_id)
    all_runs = runstate.get_all()

    if config.metric is not None:
        metric_values = np.array([run.summary_metrics.get(config.metric.name) for run in all_runs])
        parameter_values = {
            param_name: np.array([run.config[param_name]["value"] for run in all_runs])
            for param_name in config.parameters.keys()
        }

        features = calculate_feature_importance_and_correlation(metric_values, parameter_values)

        # Create the table
        table = Table(title=f"Feature Importance and Correlation for {config.metric.name}", show_lines=True)

        # Add columns
        table.add_column("Parameter", style="bold magenta")
        table.add_column("Importance", justify="right")
        table.add_column("Permutation Importance", justify="right")
        table.add_column("Pearson", justify="right")
        table.add_column("Spearman", justify="right")

        # Add rows
        for param, stats in features.items():
            table.add_row(
                param,
                f"{stats['importance']:.4f}",
                f"{stats['permutation_importance']:.4f}",
                f"{stats['pearson']:.4f}",
                f"{stats['spearman']:.4f}",
            )

        # Print using rich console
        console = Console()
        console.print(table)

        data = pd.DataFrame(
            {
                "created": [current_time_convert(run.start_time) for run in all_runs],
                config.metric.name: [run.summary_metrics.get(config.metric.name) for run in all_runs],
            }
        )

        mlflow.set_experiment(experiment_id=sweep.info.experiment_id)
        mlflow.start_run(run_id=sweep.info.run_id)

        fig = plot_metric_vs_time(data, time_col="created", metric_col=config.metric.name)
        fig.write_html("metric_vs_time.html")
        mlflow.log_artifact("metric_vs_time.html")
        Path("metric_vs_time.html").unlink(missing_ok=True)

        fig = plot_parameter_importance_and_correlation(features, metric_name=config.metric.name)
        fig.write_html("parameter_importance_and_correlation.html")
        mlflow.log_artifact("parameter_importance_and_correlation.html")
        Path("parameter_importance_and_correlation.html").unlink(missing_ok=True)
