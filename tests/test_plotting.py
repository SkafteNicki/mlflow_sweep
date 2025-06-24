import pandas as pd
import pytest
from plotly.graph_objects import Figure

from mlflow_sweep.plotting import (
    plot_metric_vs_time,
    plot_parameter_importance_and_correlation,
    plot_trial_timeline,
)


@pytest.fixture
def sample_metric_data():
    """Sample data for testing metric vs time plot."""
    data = {
        "created": [
            "2025-02-08 09:45:00",
            "2025-02-08 09:46:00",
            "2025-02-08 09:47:00",
            "2025-02-08 09:48:00",
        ],
        "accuracy": [0.75, 0.8, 0.85, 0.9],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_importance_data():
    """Sample data for testing parameter importance and correlation plot."""
    return {
        "learning_rate": {
            "importance": 0.7,
            "permutation_importance": 0.65,
            "pearson": 0.3,
            "spearman": 0.4,
        },
        "batch_size": {
            "importance": 0.3,
            "permutation_importance": 0.35,
            "pearson": -0.2,
            "spearman": -0.1,
        },
    }


@pytest.fixture
def sample_timeline_data():
    """Sample data for testing timeline plot."""
    data = {
        "start": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00"],
        "end": ["2023-01-01 10:30:00", "2023-01-01 11:30:00", "2023-01-01 12:30:00"],
        "run": ["Run 1", "Run 2", "Run 3"],
        "status": ["finished", "failed", "pruned"],
    }
    return pd.DataFrame(data)


class TestPlotting:
    def test_plot_metric_vs_time(self, sample_metric_data):
        """Test that plot_metric_vs_time returns a Figure object with expected traces."""
        fig = plot_metric_vs_time(sample_metric_data)

        # Verify the result is a Figure object
        assert isinstance(fig, Figure)

        # Verify that the figure has 2 traces (scatter points and best-so-far line)
        assert len(fig.data) == 2

        # Verify that the second trace is named 'Best so far'
        assert fig.data[1].name == "Best so far"

        # Check that the x-axis title is correctly set
        assert fig.layout.xaxis.title.text == "Created"

        # Check that the y-axis title is correctly set
        assert fig.layout.yaxis.title.text == "Accuracy"

    def test_plot_parameter_importance_and_correlation(self, sample_importance_data):
        """Test that plot_parameter_importance_and_correlation returns a Figure object with correct subplots."""
        fig = plot_parameter_importance_and_correlation(sample_importance_data, metric_name="test_metric")

        # Verify the result is a Figure object
        assert isinstance(fig, Figure)

        # Verify that the figure has 4 traces (4 bar plots in the 2x2 grid)
        assert len(fig.data) == 4

        # Check that the title contains the metric name
        assert "test_metric" in fig.layout.title.text

        # Check that the subplot titles are correctly set
        assert "Parameter Importance" in fig.layout.annotations[0].text
        assert "Pearson Correlation" in fig.layout.annotations[1].text
        assert "Spearman Correlation" in fig.layout.annotations[2].text
        assert "Permutation Importance" in fig.layout.annotations[3].text

    def test_plot_trial_timeline(self, sample_timeline_data):
        """Test that plot_trial_timeline returns a Figure object with correct data."""
        fig = plot_trial_timeline(sample_timeline_data, title="Test Timeline")

        # Verify the result is a Figure object
        assert isinstance(fig, Figure)

        # Verify that the figure has traces for each run (3 in this case)
        assert len(fig.data) == 3

        # Check that the title is correctly set
        assert fig.layout.title.text == "Test Timeline"

        # Check that the x-axis title is correctly set
        assert fig.layout.xaxis.title.text == "Datetime"

        # Check that the y-axis title is correctly set
        assert fig.layout.yaxis.title.text == "Trial"

    def test_plot_trial_timeline_custom_color_map(self, sample_timeline_data):
        """Test that plot_trial_timeline correctly uses a custom color map."""
        custom_colors = {"finished": "green", "failed": "black", "pruned": "yellow"}
        fig = plot_trial_timeline(sample_timeline_data, color_map=custom_colors)

        # Custom colors should be applied to the traces
        # This is more difficult to test directly since the colors are applied in px.timeline
        # but we can verify the function runs without errors with a custom color map
        assert isinstance(fig, Figure)
