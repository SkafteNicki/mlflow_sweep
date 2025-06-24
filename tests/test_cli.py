import tempfile
from pathlib import Path
from unittest.mock import patch

import click
import pytest
import yaml
from click.testing import CliRunner


@pytest.fixture
def cli_runner():
    """Create a CLI runner for testing Click commands."""
    return CliRunner()


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as temp:
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
        yaml.dump(config, temp)
        temp_path = temp.name

    yield temp_path

    # Cleanup
    if Path(temp_path).exists():
        Path(temp_path).unlink()


@pytest.fixture
def mock_sweep_group():
    """Create a mock sweep command group for testing."""

    @click.group(help="MLflow Sweep CLI commands.")
    def sweep():
        """MLflow Sweep CLI commands."""
        pass

    @sweep.command("init")
    @click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
    def init(config_path):
        """Initialize a new sweep configuration."""
        from mlflow_sweep.commands import init_command

        init_command(config_path)

    @sweep.command("run")
    @click.option(
        "--sweep-id",
        default="",
        type=str,
        help="ID of the sweep to run (optional if not specified will use the most recent initialized sweep)",
    )
    def run(sweep_id):
        """Start a sweep agent."""
        from mlflow_sweep.commands import run_command

        run_command(sweep_id)

    @sweep.command("finalize")
    @click.option(
        "--sweep-id",
        default="",
        type=str,
        help="ID of the sweep to finalize (optional if not specified will use the most recent initialized sweep)",
    )
    def finalize(sweep_id):
        """Finalize a sweep."""
        from mlflow_sweep.commands import finalize_command

        finalize_command(sweep_id)

    return sweep


class TestCLI:
    @patch("mlflow_sweep.commands.init_command")
    def test_init_command(self, mock_init_command, cli_runner, mock_sweep_group, temp_config_file):
        """Test that the init command calls the init_command function with correct arguments."""
        # Run the CLI command with the temporary config file
        result = cli_runner.invoke(mock_sweep_group, ["init", temp_config_file])

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Verify the init_command was called with correct arguments
        mock_init_command.assert_called_once_with(temp_config_file)

    @patch("mlflow_sweep.commands.run_command")
    def test_run_command_without_sweep_id(self, mock_run_command, cli_runner, mock_sweep_group):
        """Test that the run command calls the run_command function with empty sweep_id."""
        # Run the CLI command without sweep_id
        result = cli_runner.invoke(mock_sweep_group, ["run"])

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Verify run_command was called with empty sweep_id
        mock_run_command.assert_called_once_with("")

    @patch("mlflow_sweep.commands.run_command")
    def test_run_command_with_sweep_id(self, mock_run_command, cli_runner, mock_sweep_group):
        """Test that the run command calls the run_command function with provided sweep_id."""
        # Run the CLI command with sweep_id
        result = cli_runner.invoke(mock_sweep_group, ["run", "--sweep-id", "test-sweep-id"])

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Verify run_command was called with provided sweep_id
        mock_run_command.assert_called_once_with("test-sweep-id")

    @patch("mlflow_sweep.commands.finalize_command")
    def test_finalize_command_without_sweep_id(self, mock_finalize_command, cli_runner, mock_sweep_group):
        """Test that the finalize command calls the finalize_command function with empty sweep_id."""
        # Run the CLI command without sweep_id
        result = cli_runner.invoke(mock_sweep_group, ["finalize"])

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Verify finalize_command was called with empty sweep_id
        mock_finalize_command.assert_called_once_with("")

    @patch("mlflow_sweep.commands.finalize_command")
    def test_finalize_command_with_sweep_id(self, mock_finalize_command, cli_runner, mock_sweep_group):
        """Test that the finalize command calls the finalize_command function with provided sweep_id."""
        # Run the CLI command with sweep_id
        result = cli_runner.invoke(mock_sweep_group, ["finalize", "--sweep-id", "test-sweep-id"])

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Verify finalize_command was called with provided sweep_id
        mock_finalize_command.assert_called_once_with("test-sweep-id")

    def test_sweep_command_help(self, cli_runner, mock_sweep_group):
        """Test that the sweep command help text is displayed correctly."""
        # Run the CLI command with --help
        result = cli_runner.invoke(mock_sweep_group, ["--help"])

        # Check that the command ran successfully
        assert result.exit_code == 0

        # Check that the help text contains information about the subcommands
        assert "MLflow Sweep CLI commands" in result.output
        assert "init" in result.output
        assert "run" in result.output
        assert "finalize" in result.output
