from pydantic import BaseModel, Field
from enum import Enum
from mlflow.utils.name_utils import _generate_random_name


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
