from mlflow.tracking.context.abstract_context import RunContextProvider
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
import os


class SweepContextProvider(RunContextProvider):
    """A context provider that checks if the current run is part of a sweep.

    If so, then it returns the parent run ID as a tag such that sweep runs are linked to their parent sweep run.
    """

    def in_context(self) -> bool:
        return bool(os.environ.get("SWEEP_PARENT_RUN_ID"))

    def tags(self) -> dict[str, str]:
        return {
            MLFLOW_PARENT_RUN_ID: os.environ.get("SWEEP_PARENT_RUN_ID"),
            "mlflow.sweepRunId": os.environ.get("SWEEP_RUN_ID"),
            "mlflow.agentId": os.environ.get("SWEEP_AGENT_ID"),
        }
