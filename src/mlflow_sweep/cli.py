from math import exp
from re import sub
import click
from mlflow.cli import cli as mlflow_cli
import mlflow
import mlflow.entities
import yaml
from pydantic import BaseModel, Field, field_validator
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message="Valid config keys have changed in V2.*")
    import sweeps as sweep_module
import os
import subprocess
import json


class MetricConfig(BaseModel):
    name: str = Field(..., description="Name of the metric to track")
    goal: str = Field(..., description="Goal for the metric (e.g., 'maximize', 'minimize')")


class SweepConfig(BaseModel):
    command: str = Field(..., description="Command to run for each sweep trial")
    experiment_name: str = Field(..., description="Name of the MLflow experiment")
    sweep_name: str = Field(..., description="Name of the sweep")
    method: str = Field("random", description="Method for the sweep (e.g., 'grid', 'random')")
    metric: MetricConfig = Field(..., description="Configuration for the metric to track")
    parameters: dict[str, dict] = Field(..., description="List of parameters to sweep over")
    run_cap: int = Field(10, description="Maximum number of runs to execute in the sweep")

    @field_validator("command", mode="after")
    @classmethod
    def validate_command(cls, command: str) -> str:
        """Ensure the ${parent_run_id} placeholder is somewhere in the command."""
        if "${parent_run_id}" not in command:
            raise ValueError("The command must contain the ${parent_run_id} placeholder.")
        return command

class SweepProcessor:
    def __init__(self, config: SweepConfig, parent_sweep: mlflow.entities.Run):
        self.config = config
        self.parent_sweep = parent_sweep

    def load_previous_runs(self):
        previous_runs_path = [
            a.path for a in mlflow.artifacts.list_artifacts(run_id=self.parent_sweep.info.run_id) 
            if a.path=="proposed_parameters.json"
        ]
        if not previous_runs_path:
            return []
        previous_runs_path = previous_runs_path[0]
        artifact_uri = self.parent_sweep.info.artifact_uri.replace("file://", "")  # Remove the 'file://' prefix
        table_path = os.path.join(artifact_uri, previous_runs_path)
        with open(table_path, 'r') as file:
            previous_runs: dict = json.load(file)
        table_data = [{previous_runs["columns"][i]: row[i] for i in range(len(row))} for row in previous_runs["data"]]
        return [sweep_module.SweepRun(
            config={k: {"value": v} for k, v in sweep.items() if k not in ("run", "parent_run_id")},
            state=sweep_module.RunState.finished,
        ) for sweep in table_data]


    def propose_next(self) -> tuple[str, dict] | None:
        previous_runs = self.load_previous_runs()
        if len(previous_runs) >= self.config.run_cap:
            return None  # Stop proposing new runs if the cap is reached
        sweep_config = sweep_module.next_run(sweep_config=self.config.model_dump(), runs=previous_runs)
        if sweep_config is None:
            return None  # Grid search is exhausted or no more runs can be proposed
        proposed_parameters = {k: v["value"] for k,v in sweep_config.config.items()}
        proposed_parameters["parent_run_id"] = self.parent_sweep.info.run_id
        command = self.replace_doller_signs(self.config.command, proposed_parameters)
        proposed_parameters["run"] = len(previous_runs) + 1  # Increment run count for this sweep
        return command, proposed_parameters
    
    @staticmethod
    def replace_doller_signs(string: str, parameters: dict) -> str:
        """Replace ${parameter} with the actual parameter values."""
        for key, value in parameters.items():
            string = sub(rf"\${{{key}}}", str(value), string)
        return string


@mlflow_cli.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
def sweep(config_path):
    """Start a sweep from a config."""
    click.echo("Hello from my custom sweep command!")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    config = SweepConfig(**config)  # validate the config

    mlflow.set_experiment(config.experiment_name)
    run = mlflow.start_run(run_name=config.sweep_name)
    mlflow.set_tag("sweep", True)
    mlflow.log_artifact(config_path)
    print(run.info.run_id)


@mlflow_cli.command()
@click.option("--run-id", default="", type=str, required=True, help="ID of the run to start the agent for")
def agent(run_id: str):
    """Start a sweep agent."""

    sweeps: list[mlflow.entities.Run] = mlflow.search_runs(
        search_all_experiments=True, filter_string="tag.sweep = 'True'", output_format="list"
    )

    if run_id:
        for sweep in sweeps:
            if sweep.info.run_id == run_id:
                break
        else:
            raise ValueError(f"No sweep found with run_id: {run_id}")
    else:
        sweep = max(sweeps, key=lambda x: x.info.start_time)  # Get the most recent sweep

    sweep_config_path = mlflow.artifacts.list_artifacts(run_id=sweep.info.run_id)[0].path
    artifact_uri = sweep.info.artifact_uri.replace("file://", "")  # Remove the 'file://' prefix
    config_file_path = os.path.join(artifact_uri, sweep_config_path)

    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)

    config = SweepConfig(**config)  
    sweep_processor = SweepProcessor(config, parent_sweep=sweep)
    
    breakpoint()

    mlflow.set_experiment(experiment_id=sweep.info.experiment_id)
    with mlflow.start_run(run_id=sweep.info.run_id) as run:
        while True:
            output = sweep_processor.propose_next()
            if output is None:
                print("No more runs can be proposed or run cap reached.")
                break
            command, data = output
            table_data = {k: [str(v)] for k, v in data.items()}
            mlflow.log_table(
                data=table_data,
                artifact_file="proposed_parameters.json",
            )

            subprocess.run(command, shell=True)


if __name__ == "__main__":
    # This will run the MLflow CLI, which now includes our custom command.
    mlflow_cli()