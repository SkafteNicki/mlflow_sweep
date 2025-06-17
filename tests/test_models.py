import pytest
from pydantic import ValidationError

from mlflow_sweep.models import MetricConfig, SweepConfig


class TestMetricConfig:
    def test_valid_metric_config(self):
        # Test that a valid MetricConfig can be created
        config = MetricConfig(name="accuracy", goal="maximize")  # ty: ignore
        assert config.name == "accuracy"
        assert config.goal == "maximize"

    def test_missing_required_fields(self):
        # Test that ValidationError is raised when required fields are missing
        with pytest.raises(ValidationError):
            MetricConfig()

        with pytest.raises(ValidationError):
            MetricConfig(name="accuracy")

        with pytest.raises(ValidationError):
            MetricConfig(goal="maximize")  # ty: ignore


class TestSweepConfig:
    def test_valid_sweep_config(self):
        # Test that a valid SweepConfig can be created
        config = SweepConfig(
            command="python train.py",
            experiment_name="test_experiment",
            sweep_name="test_sweep",
            metric=MetricConfig(name="accuracy", goal="maximize"),  # ty: ignore
            parameters={"learning_rate": {"type": "float", "min": 0.001, "max": 0.1}},
        )
        assert config.command == "python train.py"
        assert config.experiment_name == "test_experiment"
        assert config.sweep_name == "test_sweep"
        assert config.method == "random"  # Default value
        assert config.metric.name == "accuracy"
        assert config.metric.goal == "maximize"
        assert config.parameters["learning_rate"]["type"] == "float"
        assert config.run_cap == 10  # Default value

    def test_missing_required_fields(self):
        # Test that ValidationError is raised when required fields are missing
        with pytest.raises(ValidationError):
            SweepConfig()

        # Missing parameters
        with pytest.raises(ValidationError):
            SweepConfig(
                command="python train.py",
                experiment_name="test_experiment",
                sweep_name="test_sweep",
                metric=MetricConfig(name="accuracy", goal="maximize"),  # ty: ignore
            )

    def test_custom_values(self):
        # Test with custom values for optional fields
        config = SweepConfig(
            command="python train.py",
            experiment_name="test_experiment",
            sweep_name="test_sweep",
            method="grid",  # ty: ignore
            metric=MetricConfig(name="loss", goal="minimize"),  # ty: ignore
            parameters={"batch_size": {"type": "int", "values": [16, 32, 64]}},
            run_cap=20,
        )
        assert config.method == "grid"
        assert config.metric.name == "loss"
        assert config.metric.goal == "minimize"
        assert config.parameters["batch_size"]["values"] == [16, 32, 64]
        assert config.run_cap == 20
