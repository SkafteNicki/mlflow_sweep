import warnings
from enum import Enum
from pathlib import Path

import yaml
from mlflow.entities import Run
from mlflow.utils.name_utils import _generate_random_name
from pydantic import BaseModel, ConfigDict, Field

with warnings.catch_warnings():
    # sweep dependency still uses V1 API of pydantic, so we need to ignore the warning about config keys
    warnings.filterwarnings("ignore", category=UserWarning, message="Valid config keys have changed in V2.*")
    from sweeps import SweepRun

    class ExtendedSweepRun(SweepRun):
        """Extended SweepRun to include additional information."""

        id: str
        start_time: int


class SweepMethodEnum(str, Enum):
    """Enumeration for sweep methods."""

    grid = "grid"
    random = "random"
    bayes = "bayes"


class GoalEnum(str, Enum):
    """Enumeration for sweep goals."""

    maximize = "maximize"
    minimize = "minimize"


class MetricConfig(BaseModel):
    """Configuration for the metric to track during the sweep.

    Attributes:
        name (str): Name of the metric to track.
        goal (GoalEnum): Goal for the metric, either 'maximize' or 'minimize'.

    """

    name: str = Field(..., description="Name of the metric to track")
    goal: GoalEnum = Field(..., description="Goal for the metric (e.g., 'maximize', 'minimize')")


class SweepConfig(BaseModel):
    """Configuration for a sweep in MLflow.

    Attributes:
        command (str): Command to run for each sweep trial.
        experiment_name (str): Name of the MLflow experiment.
        sweep_name (str): Name of the sweep, generated if not provided.
        method (SweepMethodEnum): Method for the sweep (e.g., 'grid', 'random').
        metric (MetricConfig | None): Configuration for the metric to track.
        parameters (dict[str, dict]): List of parameters to sweep over.
        run_cap (int): Maximum number of runs to execute in the sweep.

    """

    model_config = ConfigDict(extra="forbid")

    command: str = Field(..., description="Command to run for each sweep trial")
    experiment_name: str = Field("Default", description="Name of the MLflow experiment")
    sweep_name: str = Field(default_factory=lambda: "sweep-" + _generate_random_name(), description="Name of the sweep")
    method: SweepMethodEnum = Field(SweepMethodEnum.random, description="Method for the sweep (e.g., 'grid', 'random')")
    metric: MetricConfig | None = Field(None, description="Configuration for the metric to track")
    parameters: dict[str, dict] = Field(..., description="List of parameters to sweep over")
    run_cap: int = Field(10, description="Maximum number of runs to execute in the sweep")

    def model_post_init(self, context):
        """Validate the sweep configuration after initialization."""
        if self.method == SweepMethodEnum.bayes and self.metric is None:
            raise ValueError("Bayesian sweeps require a metric configuration.")

    @classmethod
    def from_sweep(cls, sweep: Run) -> "SweepConfig":
        """Create a SweepConfig instance from an MLflow Run object."""
        artifact_uri = sweep.info.artifact_uri.replace("file://", "")
        config_file_path = Path(artifact_uri) / "sweep_config.yaml"

        with Path.open(config_file_path) as file:
            config = yaml.safe_load(file)

        return cls(**config)  # Validate the config


class MetricHistory(BaseModel):
    run_id: str = Field(..., description="Run IDs associated with the metric history")
    metrics: list[dict] = Field(..., description="List of metric dicts for the run")
