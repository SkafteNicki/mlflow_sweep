import json
import pytest
from unittest.mock import MagicMock, patch, mock_open
import sweeps

from mlflow_sweep.models import SweepConfig
from mlflow_sweep.sampler import SweepProcessor


@pytest.fixture
def mock_parent_run():
    mock_run = MagicMock()
    mock_run.info.run_id = "test_run_id"
    mock_run.info.artifact_uri = "file:///tmp/artifacts"
    return mock_run


@pytest.fixture
def sweep_config():
    return SweepConfig(
        method="grid",  # ty: ignore
        metric={"name": "accuracy", "goal": "maximize"},  # ty: ignore
        parameters={"learning_rate": {"values": [0.01, 0.1]}, "batch_size": {"values": [32, 64]}},
        command="python train.py --lr=${learning_rate} --batch=${batch_size}",
        run_cap=4,
    )


class TestSweepProcessor:
    def test_init(self, sweep_config, mock_parent_run):
        processor = SweepProcessor(sweep_config, mock_parent_run)
        assert processor.config == sweep_config
        assert processor.parent_sweep == mock_parent_run

    @patch("mlflow.artifacts.list_artifacts")
    def test_load_previous_runs_no_data(self, mock_list_artifacts, mock_parent_run, sweep_config):
        mock_list_artifacts.return_value = []

        processor = SweepProcessor(sweep_config, mock_parent_run)
        result = processor.load_previous_runs()

        assert result == []
        mock_list_artifacts.assert_called_once_with(run_id="test_run_id")

    @patch("mlflow.artifacts.list_artifacts")
    @patch("os.path.join")
    @patch("builtins.open", new_callable=mock_open)
    def test_load_previous_runs_with_data(
        self, mock_file, mock_join, mock_list_artifacts, mock_parent_run, sweep_config
    ):
        # Create a mock artifact
        mock_artifact = MagicMock()
        mock_artifact.path = "proposed_parameters.json"
        mock_list_artifacts.return_value = [mock_artifact]

        # Set up the path join
        mock_join.return_value = "/tmp/artifacts/proposed_parameters.json"

        # Mock the file contents
        mock_data = {
            "columns": ["learning_rate", "batch_size", "run", "parent_run_id"],
            "data": [[0.01, 32, 1, "test_run_id"], [0.1, 64, 2, "test_run_id"]],
        }
        mock_file.return_value.__enter__.return_value.read.return_value = json.dumps(mock_data)

        processor = SweepProcessor(sweep_config, mock_parent_run)
        result = processor.load_previous_runs()

        assert len(result) == 2
        assert isinstance(result[0], sweeps.SweepRun)
        assert result[0].config["learning_rate"]["value"] == 0.01
        assert result[0].config["batch_size"]["value"] == 32
        assert result[0].state == sweeps.RunState.finished

    @patch.object(SweepProcessor, "load_previous_runs")
    @patch("sweeps.next_run")
    def test_propose_next_under_cap(self, mock_next_run, mock_load_previous, sweep_config, mock_parent_run):
        mock_load_previous.return_value = []

        # Create a mock sweep configuration
        mock_config = MagicMock()
        mock_config.config = {"learning_rate": {"value": 0.01}, "batch_size": {"value": 32}}
        mock_next_run.return_value = mock_config

        processor = SweepProcessor(sweep_config, mock_parent_run)
        command, params = processor.propose_next()  # ty: ignore

        assert command == "python train.py --lr=0.01 --batch=32"
        assert params == {"learning_rate": 0.01, "batch_size": 32, "run": 1}
        mock_next_run.assert_called_once_with(sweep_config=sweep_config.model_dump(), runs=[])

    @patch.object(SweepProcessor, "load_previous_runs")
    def test_propose_next_at_cap(self, mock_load_previous, sweep_config, mock_parent_run):
        # Create 4 mock runs (equal to run cap)
        mock_runs = [MagicMock() for _ in range(4)]
        mock_load_previous.return_value = mock_runs

        processor = SweepProcessor(sweep_config, mock_parent_run)
        result = processor.propose_next()

        assert result is None  # Should return None when run cap is reached

    def test_replace_dollar_signs(self):
        parameters = {"learning_rate": 0.01, "batch_size": 32}
        template = "python train.py --lr=${learning_rate} --batch=${batch_size}"
        expected = "python train.py --lr=0.01 --batch=32"

        result = SweepProcessor.replace_dollar_signs(template, parameters)
        assert result == expected
