import os
from enum import Enum

import yaml
from mlflow.entities import Run
from mlflow.utils.name_utils import _generate_random_name
from pydantic import BaseModel, ConfigDict, Field


class SweepMethodEnum(str, Enum):
    grid = "grid"
    random = "random"


class GoalEnum(str, Enum):
    maximize = "maximize"
    minimize = "minimize"


class MetricConfig(BaseModel):
    name: str = Field(..., description="Name of the metric to track")
    goal: GoalEnum = Field(..., description="Goal for the metric (e.g., 'maximize', 'minimize')")


class SweepConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str = Field(..., description="Command to run for each sweep trial")
    experiment_name: str = Field("", description="Name of the MLflow experiment")
    sweep_name: str = Field("", description="Name of the sweep")
    method: SweepMethodEnum = Field(SweepMethodEnum.random, description="Method for the sweep (e.g., 'grid', 'random')")
    metric: MetricConfig | None = Field(None, description="Configuration for the metric to track")
    parameters: dict[str, dict] = Field(..., description="List of parameters to sweep over")
    run_cap: int = Field(10, description="Maximum number of runs to execute in the sweep")

    def model_post_init(self, context):
        """Post-initialization hook to set default values if not provided."""
        if self.experiment_name == "":
            self.experiment_name = "Default"
        if self.sweep_name == "":
            self.sweep_name = "sweep-" + _generate_random_name()

    @classmethod
    def from_sweep(cls, sweep: Run) -> "SweepConfig":
        """Create a SweepConfig instance from an MLflow Run object."""
        artifact_uri = sweep.info.artifact_uri.replace("file://", "")
        config_file_path = os.path.join(artifact_uri, "sweep_config.yaml")

        with open(config_file_path) as file:
            config = yaml.safe_load(file)

        return cls(**config)  # Validate the config
