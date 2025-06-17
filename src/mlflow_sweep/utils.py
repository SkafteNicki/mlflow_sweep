import datetime

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance


def calculate_feature_importance_and_correlation(
    metric_value: np.ndarray, parameter_values: dict[str, np.ndarray]
) -> dict:
    """Calculate feature importance and correlation coefficients for hyperparameters.

    Args:
        metric_value (np.ndarray): Array of metric values (e.g., validation loss).
        parameter_values (dict[str, np.ndarray]): Dictionary where keys are parameter names and values are arrays of parameter values.

    Returns:
        dict: Dictionary with parameter names as keys and dictionaries containing importance,
              permutation importance, and correlation metrics as values.

    Examples:
        >>> import numpy as np
        >>> np.random.seed(42)  # For reproducibility
        >>> metric = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
        >>> params = {
        ...     'learning_rate': np.array([0.01, 0.02, 0.01, 0.03, 0.02]),
        ...     'batch_size': np.array([32, 64, 32, 128, 64])
        ... }
        >>> result = calculate_feature_importance_and_correlation(metric, params)
        >>> sorted(result.keys())
        ['batch_size', 'learning_rate']
        >>> all(k in result['learning_rate'] for k in ['importance', 'permutation_importance', 'correlation'])
        True
        >>> all(k in result['learning_rate']['correlation'] for k in ['pearson', 'spearman'])
        True
    """
    # Check for empty parameter set
    if not parameter_values:
        raise ValueError("Parameter values dictionary cannot be empty")

    # Check that all parameter arrays have the same length as the metric array
    for param_name, param_array in parameter_values.items():
        if len(param_array) != len(metric_value):
            raise ValueError(
                f"Parameter '{param_name}' has {len(param_array)} values, but metric has {len(metric_value)} values"
            )

    data = np.column_stack(list(parameter_values.values()))

    model = RandomForestRegressor()
    model.fit(data, metric_value)
    importances = model.feature_importances_

    perm = permutation_importance(model, data, metric_value, n_repeats=30)
    perm_importances = perm["importances_mean"]

    correlations = {}
    for i, param in enumerate(parameter_values.keys()):
        pearson_corr, _ = pearsonr(data[:, i], metric_value)
        spearman_corr, _ = spearmanr(data[:, i], metric_value)
        correlations[param] = {"pearson": pearson_corr, "spearman": spearman_corr}

    result = {
        k: {
            "importance": float(v),
            "permutation_importance": float(perm_importances[i]),
            "pearson": float(correlations[k]["pearson"]),
            "spearman": float(correlations[k]["spearman"]),
        }
        for i, (k, v) in enumerate(zip(parameter_values.keys(), importances))
    }
    return result


def current_time_convert(ts_ms: int) -> str:
    """Convert a timestamp in milliseconds to a formatted UTC string.

    Args:
        ts_ms (int): Timestamp in milliseconds.

    Returns:
        str: Formatted UTC time string in 'YYYY-MM-DD HH:MM:SS' format.
    """
    dt_utc = datetime.datetime.utcfromtimestamp(ts_ms / 1000)
    return dt_utc.strftime("%Y-%m-%d %H:%M:%S")
