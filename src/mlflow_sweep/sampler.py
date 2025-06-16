from mlflow_sweep.models import SweepConfig
from mlflow.entities import Run
import mlflow
import json
import os
from re import sub

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message="Valid config keys have changed in V2.*")
    import sweeps as sweep_module


class SweepProcessor:
    def __init__(self, config: SweepConfig, parent_sweep: Run):
        self.config = config
        self.parent_sweep = parent_sweep

    def load_previous_runs(self):
        previous_runs_path = [
            a.path
            for a in mlflow.artifacts.list_artifacts(run_id=self.parent_sweep.info.run_id)
            if a.path == "proposed_parameters.json"
        ]
        if not previous_runs_path:
            return []
        previous_runs_path = previous_runs_path[0]
        artifact_uri = self.parent_sweep.info.artifact_uri.replace("file://", "")  # Remove the 'file://' prefix
        table_path = os.path.join(artifact_uri, previous_runs_path)
        with open(table_path, "r") as file:
            previous_runs: dict = json.load(file)
        table_data = [{previous_runs["columns"][i]: row[i] for i in range(len(row))} for row in previous_runs["data"]]
        return [
            sweep_module.SweepRun(
                config={k: {"value": v} for k, v in sweep.items() if k not in ("run", "parent_run_id")},
                state=sweep_module.RunState.finished,
            )
            for sweep in table_data
        ]

    def propose_next(self) -> tuple[str, dict] | None:
        previous_runs = self.load_previous_runs()
        if len(previous_runs) >= self.config.run_cap:
            return None  # Stop proposing new runs if the cap is reached
        sweep_config = sweep_module.next_run(sweep_config=self.config.model_dump(), runs=previous_runs)
        if sweep_config is None:
            return None  # Grid search is exhausted or no more runs can be proposed
        proposed_parameters = {k: v["value"] for k, v in sweep_config.config.items()}
        command = self.replace_dollar_signs(self.config.command, proposed_parameters)
        proposed_parameters["run"] = len(previous_runs) + 1  # Increment run count for this sweep
        return command, proposed_parameters

    @staticmethod
    def replace_dollar_signs(string: str, parameters: dict) -> str:
        """Replace ${parameter} with the actual parameter values."""
        for key, value in parameters.items():
            string = sub(rf"\${{{key}}}", str(value), string)
        return string
