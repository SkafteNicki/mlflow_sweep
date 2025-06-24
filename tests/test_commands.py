import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from mlflow.entities import Run, RunData, RunInfo

from mlflow_sweep.commands import determine_sweep, finalize_command, init_command, run_command
from mlflow_sweep.models import SweepConfig


@pytest.fixture
def mock_run():
    """Create a mock MLflow Run object."""
    run_info = MagicMock(spec=RunInfo)
    run_info.run_id = "test-run-id"
    run_info.experiment_id = "test-experiment-id"
    run_info.artifact_uri = "file:///path/to/artifacts"
    run_info.start_time = 1640995200000  # 2022-01-01

    run_data = MagicMock(spec=RunData)
    run_data.tags = {"sweep": "True"}

    mock_run = MagicMock(spec=Run)
    mock_run.info = run_info
    mock_run.data = run_data

    return mock_run


@pytest.fixture
def temp_config_file(tmp_path):
    """Create a temporary config file for testing."""
    config = {
        "command": "python train.py --lr=${learning_rate} --batch=${batch_size}",
        "experiment_name": "test-experiment",
        "sweep_name": "test-sweep",
        "method": "random",
        "metric": {"name": "accuracy", "goal": "maximize"},
        "parameters": {
            "learning_rate": {"distribution": "uniform", "min": 0.001, "max": 0.1},
            "batch_size": {"values": [16, 32, 64]},
        },
        "run_cap": 5,
    }

    config_file = tmp_path / "test_sweep_config.yaml"
    with Path(config_file).open("w") as f:
        yaml.dump(config, f)

    return config_file


class TestCommands:
    @patch("mlflow.search_runs")
    def test_determine_sweep_with_id(self, mock_search_runs, mock_run):
        """Test determine_sweep when a sweep_id is provided."""
        mock_search_runs.return_value = [mock_run]

        result = determine_sweep("test-run-id")

        mock_search_runs.assert_called_once_with(
            search_all_experiments=True, filter_string="tag.sweep = 'True'", output_format="list"
        )

        assert result == mock_run

    @patch("mlflow.search_runs")
    def test_determine_sweep_without_id(self, mock_search_runs, mock_run):
        """Test determine_sweep when no sweep_id is provided (use most recent)."""
        run1 = MagicMock(spec=Run)
        run1.info.start_time = 1609459200000  # 2021-01-01

        run2 = mock_run  # 2022-01-01 (more recent)

        mock_search_runs.return_value = [run1, run2]

        result = determine_sweep("")

        mock_search_runs.assert_called_once_with(
            search_all_experiments=True, filter_string="tag.sweep = 'True'", output_format="list"
        )

        assert result == run2  # Should select the most recent sweep

    @patch("mlflow.search_runs")
    def test_determine_sweep_with_invalid_id(self, mock_search_runs):
        """Test determine_sweep with an invalid sweep_id."""
        mock_search_runs.return_value = []

        with pytest.raises(ValueError, match="No sweep found with sweep_id: invalid-id"):
            determine_sweep("invalid-id")

    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    @patch("mlflow.set_tag")
    @patch("mlflow.log_artifact")
    def test_init_command(
        self, mock_log_artifact, mock_set_tag, mock_start_run, mock_set_experiment, temp_config_file, mock_run
    ):
        """Test init_command with a valid config file."""
        mock_start_run.return_value = mock_run

        init_command(temp_config_file)

        mock_set_experiment.assert_called_once_with("test-experiment")
        mock_start_run.assert_called_once_with(run_name="test-sweep")
        mock_set_tag.assert_called_once_with("sweep", True)
        mock_log_artifact.assert_called_once()  # The path will be a temp file

    @patch("mlflow_sweep.commands.determine_sweep")
    @patch("mlflow_sweep.commands.SweepConfig.from_sweep")
    @patch("mlflow_sweep.commands.SweepState")
    @patch("mlflow_sweep.commands.SweepSampler")
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    @patch("subprocess.run")
    @patch.dict(os.environ, {}, clear=True)
    def test_run_command(
        self,
        mock_subprocess,
        mock_start_run,
        mock_set_experiment,
        mock_sweep_sampler,
        mock_sweep_state,
        mock_from_sweep,
        mock_determine_sweep,
        mock_run,
    ):
        """Test run_command."""
        # Setup mocks
        mock_determine_sweep.return_value = mock_run

        config = SweepConfig(
            command="python train.py --lr=${learning_rate} --batch=${batch_size}",
            parameters={"learning_rate": {"distribution": "uniform", "min": 0.001, "max": 0.1}},
        )
        mock_from_sweep.return_value = config

        mock_sampler_instance = MagicMock()
        mock_sampler_instance.propose_next.side_effect = [
            (
                "python train.py --lr=0.01 --batch=32",
                {"learning_rate": 0.01, "batch_size": 32, "run": 1, "sweep_run_id": "run-id-1"},
            ),
            None,  # Indicate we're done after one run
        ]
        mock_sweep_sampler.return_value = mock_sampler_instance

        # Run the command
        run_command("test-run-id")

        # Verify method calls
        mock_determine_sweep.assert_called_once_with("test-run-id")
        mock_from_sweep.assert_called_once_with(mock_run)
        mock_set_experiment.assert_called_once_with(experiment_id="test-experiment-id")
        mock_start_run.assert_called_once_with(run_id="test-run-id")

        # Verify environment variables in subprocess call
        subprocess_env = mock_subprocess.call_args[1]["env"]
        assert "SWEEP_PARENT_RUN_ID" in subprocess_env
        assert "SWEEP_AGENT_ID" in subprocess_env
        assert "SWEEP_RUN_ID" in subprocess_env

        # Verify subprocess was called with expected command
        mock_subprocess.assert_called_once()
        assert mock_subprocess.call_args[0][0] == "python train.py --lr=0.01 --batch=32"

    @patch("mlflow_sweep.commands.determine_sweep")
    @patch("mlflow_sweep.commands.SweepConfig.from_sweep")
    @patch("mlflow_sweep.commands.SweepState")
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    @patch("mlflow.log_artifact")
    @patch("mlflow_sweep.commands.plot_trial_timeline")
    @patch("mlflow_sweep.commands.plot_metric_vs_time")
    @patch("mlflow_sweep.commands.plot_parameter_importance_and_correlation")
    @patch("mlflow_sweep.commands.calculate_feature_importance_and_correlation")
    def test_finalize_command(
        self,
        mock_calculate,
        mock_param_plot,
        mock_metric_plot,
        mock_timeline,
        mock_log_artifact,
        mock_start_run,
        mock_set_experiment,
        mock_sweep_state,
        mock_from_sweep,
        mock_determine_sweep,
        mock_run,
    ):
        """Test finalize_command."""
        # Setup mocks
        mock_determine_sweep.return_value = mock_run

        config = SweepConfig(
            command="python train.py",
            metric={"name": "accuracy", "goal": "maximize"},  # ty: ignore
            parameters={"learning_rate": {"distribution": "uniform", "min": 0.001, "max": 0.1}},
        )
        mock_from_sweep.return_value = config

        # Setup SweepState mock
        state_instance = MagicMock()
        run1 = MagicMock()
        run1.id = "run1"
        run1.start_time = 1609459200000  # 2021-01-01
        run1.end_time = 1609462800000  # 2021-01-01 + 1 hour
        run1.state = "FINISHED"
        run1.summary_metrics = {"accuracy": 0.85}
        run1.config = {"learning_rate": {"value": 0.01}}

        state_instance.get_all.return_value = [run1]
        mock_sweep_state.return_value = state_instance

        # Setup figure mocks
        mock_timeline.return_value = MagicMock()
        mock_metric_plot.return_value = MagicMock()
        mock_param_plot.return_value = MagicMock()

        # Setup importance calculation mock
        mock_calculate.return_value = {
            "learning_rate": {"importance": 1.0, "permutation_importance": 1.0, "pearson": 0.5, "spearman": 0.5}
        }

        # Run the command
        with patch("pathlib.Path.unlink"):  # Mock file deletion
            finalize_command("test-run-id")

        # Verify method calls
        mock_determine_sweep.assert_called_once_with("test-run-id")
        mock_from_sweep.assert_called_once_with(mock_run)
        mock_set_experiment.assert_called_once_with(experiment_id="test-experiment-id")
        mock_start_run.assert_called_once_with(run_id="test-run-id")

        # Verify plots were created and logged
        assert mock_timeline.call_count == 1
        assert mock_calculate.call_count == 1
        assert mock_metric_plot.call_count == 1
        assert mock_param_plot.call_count == 1
        assert mock_log_artifact.call_count == 3  # Three plots should be logged
