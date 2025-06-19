from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from mlflow_sweep.models import GoalEnum, MetricConfig, SweepConfig, SweepMethodEnum


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

    def test_default_experiment_and_sweep_names(self):
        # Test that experiment_name and sweep_name get default values if not provided
        with patch("mlflow_sweep.models._generate_random_name", return_value="random-name"):
            config = SweepConfig(
                command="python train.py",
                parameters={"learning_rate": {"type": "float", "min": 0.001, "max": 0.1}},
            )
            assert config.experiment_name == "Default"
            assert config.sweep_name == "sweep-random-name"

    def test_enum_values(self):
        # Test that enum values are correctly validated
        config = SweepConfig(
            command="python train.py",
            method=SweepMethodEnum.grid,
            metric=MetricConfig(name="accuracy", goal=GoalEnum.maximize),
            parameters={"learning_rate": {"type": "float", "min": 0.001, "max": 0.1}},
        )
        assert config.method == SweepMethodEnum.grid
        assert config.metric.goal == GoalEnum.maximize

    @patch("pathlib.Path.open")
    def test_from_sweep(self, mock_open):
        # Test the from_sweep class method
        # Mock the MLflow Run object
        mock_run = MagicMock()
        mock_run.info.artifact_uri = "file:///path/to/artifacts"

        # Mock the YAML file content
        mock_config = {
            "command": "python train.py",
            "experiment_name": "test_experiment",
            "sweep_name": "test_sweep",
            "method": "grid",
            "metric": {"name": "accuracy", "goal": "maximize"},
            "parameters": {"learning_rate": {"type": "float", "min": 0.001, "max": 0.1}},
            "run_cap": 15,
        }

        # Configure the mock to return the YAML data
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        mock_yaml = MagicMock()
        mock_yaml.return_value = mock_config

        with patch("yaml.safe_load", mock_yaml):
            config = SweepConfig.from_sweep(mock_run)

            # Assert that the correct file was opened
            mock_open.assert_called_once_with(Path("/path/to/artifacts/sweep_config.yaml"))

            # Assert that the config was loaded correctly
            assert config.command == "python train.py"
            assert config.experiment_name == "test_experiment"
            assert config.sweep_name == "test_sweep"
            assert config.method == "grid"
            assert config.metric.name == "accuracy"
            assert config.metric.goal == "maximize"
            assert config.parameters["learning_rate"]["type"] == "float"
            assert config.run_cap == 15
