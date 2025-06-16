import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from scipy.stats import pearsonr, spearmanr


def calculate_feature_importance_and_correlation(
    metric_value: np.ndarray, parameter_values: dict[str, np.ndarray]
) -> dict:
    """Calculate feature importance and correlation coefficients for hyperparameters.

    Args:
        metric_value (np.ndarray): Array of metric values (e.g., validation loss).
        parameter_values (dict[str, np.ndarray]): Dictionary where keys are parameter names and values are arrays of parameter values.
    """
    data = np.column_stack(list(parameter_values.values()))

    model = RandomForestRegressor()
    model.fit(data, metric_value)
    importances = model.feature_importances_

    perm = permutation_importance(model, data, metric_value, n_repeats=30)
    perm_importances = perm.importances_mean

    correlations = {}
    for i, param in enumerate(parameter_values.keys()):
        pearson_corr, _ = pearsonr(data[:, i], metric_value)
        spearman_corr, _ = spearmanr(data[:, i], metric_value)
        correlations[param] = {"pearson": pearson_corr, "spearman": spearman_corr}

    result = {
        k: {"importance": v, "permutation_importance": perm_importances[i], "correlation": correlations[k]}
        for i, (k, v) in enumerate(zip(parameter_values.keys(), importances))
    }
    return result
