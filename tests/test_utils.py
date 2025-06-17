import numpy as np
import pytest

from mlflow_sweep.utils import calculate_feature_importance_and_correlation


@pytest.fixture
def sample_data():
    """Fixture providing sample data for testing."""
    metric = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
    params = {
        "learning_rate": np.array([0.01, 0.02, 0.01, 0.03, 0.02]),
        "batch_size": np.array([32, 64, 32, 128, 64]),
        "n_estimators": np.array([100, 200, 150, 300, 250]),
    }
    return metric, params


def test_calculate_feature_importance_and_correlation_basic(sample_data):
    """Test basic functionality of the feature importance calculation."""
    metric, params = sample_data
    result = calculate_feature_importance_and_correlation(metric, params)

    # Check if all parameters are in the result
    assert set(result.keys()) == set(params.keys())

    # Check if all expected keys exist in the result for each parameter
    for param in params:
        assert "importance" in result[param]
        assert "permutation_importance" in result[param]
        assert "correlation" in result[param]
        assert "pearson" in result[param]["correlation"]
        assert "spearman" in result[param]["correlation"]


def test_calculate_feature_importance_values(sample_data):
    """Test that importance values are in the expected range."""
    metric, params = sample_data
    result = calculate_feature_importance_and_correlation(metric, params)

    # Importance values from RandomForest should sum to 1
    total_importance = sum(result[param]["importance"] for param in params)
    assert np.isclose(total_importance, 1.0, atol=1e-10)

    # Check that all importance values are non-negative
    for param in params:
        assert result[param]["importance"] >= 0
        assert result[param]["permutation_importance"] >= 0


def test_calculate_feature_importance_correlations(sample_data):
    """Test that correlation values are in the correct range."""
    metric, params = sample_data
    result = calculate_feature_importance_and_correlation(metric, params)

    # Correlation values should be between -1 and 1
    for param in params:
        assert -1 <= result[param]["correlation"]["pearson"] <= 1
        assert -1 <= result[param]["correlation"]["spearman"] <= 1


def test_empty_parameter_set():
    """Test behavior with empty parameter set."""
    metric = np.array([0.1, 0.2, 0.3])
    params = {}

    with pytest.raises(ValueError):
        calculate_feature_importance_and_correlation(metric, params)


def test_single_parameter():
    """Test with a single parameter."""
    metric = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
    params = {"learning_rate": np.array([0.01, 0.02, 0.01, 0.03, 0.02])}

    result = calculate_feature_importance_and_correlation(metric, params)

    assert "learning_rate" in result
    assert result["learning_rate"]["importance"] == 1.0  # Single feature should have importance 1


def test_different_length_arrays():
    """Test that arrays of different lengths raise an error."""
    metric = np.array([0.1, 0.2, 0.3])
    params = {
        "learning_rate": np.array([0.01, 0.02]),  # Only 2 values
        "batch_size": np.array([32, 64, 128]),  # 3 values
    }

    with pytest.raises(ValueError):
        calculate_feature_importance_and_correlation(metric, params)


def test_with_constant_parameter():
    """Test behavior when a parameter has constant values."""
    metric = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
    params = {
        "learning_rate": np.array([0.01, 0.02, 0.01, 0.03, 0.02]),
        "constant_param": np.array([1.0, 1.0, 1.0, 1.0, 1.0]),  # Constant parameter
    }

    result = calculate_feature_importance_and_correlation(metric, params)

    # Constant features should have zero importance in RandomForest
    assert result["constant_param"]["importance"] == 0.0
