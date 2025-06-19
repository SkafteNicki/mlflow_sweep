import uuid
from unittest.mock import MagicMock, patch

import pytest

from mlflow_sweep.models import SweepConfig
from mlflow_sweep.sampler import SweepSampler
from mlflow_sweep.sweepstate import SweepState


@pytest.fixture
def mock_sweepstate():
    mock_state = MagicMock(spec=SweepState)
    mock_state.get_all.return_value = []
    return mock_state


@pytest.fixture
def sweep_config():
    return SweepConfig(
        method="grid",  # ty: ignore
        metric={"name": "accuracy", "goal": "maximize"},  # ty: ignore
        parameters={"learning_rate": {"values": [0.01, 0.1]}, "batch_size": {"values": [32, 64]}},
        command="python train.py --lr=${learning_rate} --batch=${batch_size}",
        run_cap=4,
    )


class TestSweepSampler:
    def test_init(self, sweep_config, mock_sweepstate):
        sampler = SweepSampler(sweep_config, mock_sweepstate)
        assert sampler.config == sweep_config
        assert sampler.sweepstate == mock_sweepstate

    def test_propose_next_under_cap(self, sweep_config, mock_sweepstate):
        # Create a mock sweep configuration
        mock_config = MagicMock()
        mock_config.config = {"learning_rate": {"value": 0.01}, "batch_size": {"value": 32}}

        with (
            patch("sweeps.next_run", return_value=mock_config),
            patch("uuid.uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")),
        ):
            sampler = SweepSampler(sweep_config, mock_sweepstate)
            command, params = sampler.propose_next()  # ty: ignore

            assert command == "python train.py --lr=0.01 --batch=32"
            assert params["learning_rate"] == 0.01
            assert params["batch_size"] == 32
            assert params["run"] == 1
            assert params["sweep_run_id"] == "12345678-1234-5678-1234-567812345678"

    def test_propose_next_at_cap(self, sweep_config, mock_sweepstate):
        # Create 4 mock runs (equal to run cap)
        mock_sweepstate.get_all.return_value = [MagicMock() for _ in range(4)]

        sampler = SweepSampler(sweep_config, mock_sweepstate)
        result = sampler.propose_next()

        assert result is None  # Should return None when run cap is reached

    def test_propose_next_exhausted(self, sweep_config, mock_sweepstate):
        # Test when the sweep module returns None (e.g., grid search exhausted)
        with patch("sweeps.next_run", return_value=None):
            sampler = SweepSampler(sweep_config, mock_sweepstate)
            result = sampler.propose_next()

            assert result is None

    def test_replace_dollar_signs(self):
        parameters = {"learning_rate": 0.01, "batch_size": 32}
        template = "python train.py --lr=${learning_rate} --batch=${batch_size}"
        expected = "python train.py --lr=0.01 --batch=32"

        result = SweepSampler.replace_dollar_signs(template, parameters)
        assert result == expected
