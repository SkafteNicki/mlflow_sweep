from unittest.mock import patch

import pytest
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

from mlflow_sweep.runcontext import SweepContextProvider


@pytest.fixture
def provider():
    return SweepContextProvider()


@patch.dict("os.environ", {"SWEEP_PARENT_RUN_ID": "parent123"})
def test_in_context_with_parent_id(provider):
    """Test that in_context returns True when SWEEP_PARENT_RUN_ID is set"""
    assert provider.in_context() is True


@patch.dict("os.environ", {})
def test_in_context_without_parent_id(provider):
    """Test that in_context returns False when SWEEP_PARENT_RUN_ID is not set"""
    assert provider.in_context() is False


@patch.dict("os.environ", {"SWEEP_PARENT_RUN_ID": "parent123", "SWEEP_RUN_ID": "run456", "SWEEP_AGENT_ID": "agent789"})
def test_tags_with_all_env_vars(provider):
    """Test that tags returns correct dictionary when all environment variables are set"""
    expected_tags = {
        MLFLOW_PARENT_RUN_ID: "parent123",
        "mlflow.sweepRunId": "run456",
        "mlflow.agentId": "agent789",
    }
    assert provider.tags() == expected_tags


@patch.dict("os.environ", {"SWEEP_PARENT_RUN_ID": "parent123"})
def test_tags_with_partial_env_vars(provider):
    """Test that tags returns correct dictionary when only parent run ID is set"""
    expected_tags = {
        MLFLOW_PARENT_RUN_ID: "parent123",
        "mlflow.sweepRunId": None,
        "mlflow.agentId": None,
    }
    assert provider.tags() == expected_tags
