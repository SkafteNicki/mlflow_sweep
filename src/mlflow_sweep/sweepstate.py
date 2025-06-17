import warnings
from mlflow import MlflowClient
from mlflow.entities import Run
import mlflow
import os
import json

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message="Valid config keys have changed in V2.*")
    from sweeps import SweepRun, RunState


def status_mapping(mlflow_status: str) -> RunState:
    """Map MLflow run status to SweepRun state."""
    if mlflow_status == "RUNNING":
        return RunState.running
    elif mlflow_status == "SCHEDULED":
        return RunState.pending
    elif mlflow_status == "FINISHED":
        return RunState.finished
    elif mlflow_status == "FAILED":
        return RunState.failed
    else:
        return RunState.killed


class ExtendedSweepRun(SweepRun):
    """Extended SweepRun to include additional information."""

    id: str
    start_time: int


class SweepState:
    def __init__(self, sweep_id: str):
        self.sweep_id = sweep_id
        self.client = MlflowClient()

    def get_all(self) -> list[ExtendedSweepRun]:
        mlflow_runs: list[Run] = mlflow.search_runs(  # ty: ignore[invalid-assignment]
            search_all_experiments=True,
            filter_string=f"tag.mlflow.parentRunId = '{self.sweep_id}'",
            output_format="list",
        )
        parameters = self.get_parameters()

        mlflow_runs_sorted = sorted(mlflow_runs, key=lambda run: run.data.tags.get("mlflow.sweepRunId"))
        parameters_sorted = sorted(parameters, key=lambda x: x["sweep_run_id"])

        return [
            self.convert_from_mlflow_runinfo_to_sweep_run(run, params)
            for run, params in zip(mlflow_runs_sorted, parameters_sorted)
        ]

    def get(self, run_id: str) -> ExtendedSweepRun:
        """Retrieve a SweepRun by its run_id."""
        mlflow_run: Run = mlflow.search_runs(  # ty: ignore[invalid-assignment]
            search_all_experiments=True,
            filter_string=f"tag.mlflow.sweepRunId = '{run_id}'",
            output_format="list",
        )[0]
        parameters = self.get_parameters()
        parameters = next((p for p in parameters if p["sweep_run_id"] == run_id), {})
        return self.convert_from_mlflow_runinfo_to_sweep_run(mlflow_run, parameters)

    def save(self, run_id: str):
        """Save the SweepRun to MLflow."""
        sweep_run = self.get(run_id)
        self.client.log_dict(
            run_id=self.sweep_id,
            dictionary=sweep_run.model_dump(),
            artifact_file=f"sweep_run_{sweep_run.id}.json",
        )

    @staticmethod
    def convert_from_mlflow_runinfo_to_sweep_run(mlflow_run: Run, params: dict) -> ExtendedSweepRun:
        """Convert an MLflow Run to a SweepRun."""
        params = {k: {"value": v} for k, v in params.items() if k not in ["run", "sweep_run_id"]}
        return ExtendedSweepRun(
            id=mlflow_run.info.run_id,
            name=mlflow_run.info.run_name,
            summaryMetrics=mlflow_run.data.metrics,  # ty: ignore[unknown-argument]
            config=params,
            state=status_mapping(mlflow_run.info.status),
            start_time=mlflow_run.info.start_time,
        )

    def get_parameters(self):
        if "proposed_parameters.json" not in [a.path for a in self.client.list_artifacts(self.sweep_id)]:
            return []
        artifact_uri = self.client.get_run(self.sweep_id).info.artifact_uri.replace("file://", "")
        table_path = os.path.join(artifact_uri, "proposed_parameters.json")
        with open(table_path, "r") as file:
            previous_runs: dict = json.load(file)
        table_data = [{previous_runs["columns"][i]: row[i] for i in range(len(row))} for row in previous_runs["data"]]
        return table_data
