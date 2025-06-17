import warnings
from mlflow import MlflowClient
from mlflow.entities import Run
import mlflow

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message="Valid config keys have changed in V2.*")
    from sweeps import SweepRun, RunState


class ExtendedSweepRun(SweepRun):
    """Extended SweepRun to include additional information."""

    id: str
    start_time: int


class SweepState:
    def __init__(self, sweep_id: str):
        self.sweep_id = sweep_id
        self.client = MlflowClient()

    def save(self, run_id: str):
        mlflow_run: Run = mlflow.search_runs(  # ty: ignore[invalid-assignment]
            search_all_experiments=True,
            filter_string=f"tag.mlflow.sweepRunId = '{run_id}'",
            output_format="list",
        )[0]
        sweep_run = self.convert_from_mlflow_runinfo_to_sweep_run(mlflow_run)

        self.client.log_dict(
            run_id=self.sweep_id,
            dictionary=sweep_run.model_dump(),
            artifact_file=f"sweep_run_{sweep_run.id}.json",
        )

    @staticmethod
    def convert_from_mlflow_runinfo_to_sweep_run(run: Run) -> ExtendedSweepRun:
        """Convert an MLflow Run to a SweepRun."""
        return ExtendedSweepRun(
            id=run.info.run_id,
            name=run.info.run_name,
            summaryMetrics=run.data.metrics,  # ty: ignore[unknown-argument]
            config=run.data.params,
            state=RunState.finished if run.info.status == "FINISHED" else RunState.failed,
            start_time=run.info.start_time,
        )
