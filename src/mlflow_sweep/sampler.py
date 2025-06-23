import uuid
import warnings
from re import sub

from sklearn.exceptions import ConvergenceWarning

from mlflow_sweep.models import SweepConfig
from mlflow_sweep.sweepstate import SweepState

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning, message="Valid config keys have changed in V2.*")
    import sweeps as sweep_module


class SweepSampler:
    """Sampler for proposing new runs in a sweep based on the provided configuration and state.

    Args:
        config (SweepConfig): The sweep configuration containing the command and parameters.
        sweepstate (SweepState): Class managing the state of the sweep, including previous runs and metrics.
    """

    def __init__(self, config: SweepConfig, sweepstate: SweepState) -> None:
        self.config = config
        self.sweepstate = sweepstate

    def propose_next(self) -> tuple[str, dict] | None:
        """Propose the next run command and parameters based on the sweep configuration and state."""
        previous_runs = self.sweepstate.get_all(with_metric=self.config.metric.name if self.config.metric else "")
        if len(previous_runs) >= self.config.run_cap:
            return None  # Stop proposing new runs if the cap is reached

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=ConvergenceWarning, message="The optimal value found for dimension 0 of parameter.*"
            )
            sweep_config = sweep_module.next_run(sweep_config=self.config.model_dump(), runs=previous_runs)

        if sweep_config is None:
            return None  # Grid search is exhausted or no more runs can be proposed
        proposed_parameters = {k: v["value"] for k, v in sweep_config.config.items()}
        command = self.replace_dollar_signs(self.config.command, proposed_parameters)
        proposed_parameters["run"] = len(previous_runs) + 1  # Increment run count for this sweep
        proposed_parameters["sweep_run_id"] = str(uuid.uuid4())  # Unique ID for this run
        return command, proposed_parameters

    @staticmethod
    def replace_dollar_signs(string: str, parameters: dict) -> str:
        """Replace ${parameter} with the actual parameter values."""
        for key, value in parameters.items():
            string = sub(rf"\${{{key}}}", str(value), string)
        return string
