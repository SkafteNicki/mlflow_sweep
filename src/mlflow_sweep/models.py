from pydantic import BaseModel, Field


class MetricConfig(BaseModel):
    name: str = Field(..., description="Name of the metric to track")
    goal: str = Field(
        ..., description="Goal for the metric (e.g., 'maximize', 'minimize')"
    )


class SweepConfig(BaseModel):
    command: str = Field(..., description="Command to run for each sweep trial")
    experiment_name: str = Field(..., description="Name of the MLflow experiment")
    sweep_name: str = Field(..., description="Name of the sweep")
    method: str = Field(
        "random", description="Method for the sweep (e.g., 'grid', 'random')"
    )
    metric: MetricConfig = Field(
        ..., description="Configuration for the metric to track"
    )
    parameters: dict[str, dict] = Field(
        ..., description="List of parameters to sweep over"
    )
    run_cap: int = Field(
        10, description="Maximum number of runs to execute in the sweep"
    )
