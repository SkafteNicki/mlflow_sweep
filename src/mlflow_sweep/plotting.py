import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_metric_vs_time(dataframe: pd.DataFrame, time_col: str = "created", metric_col: str = "accuracy") -> go.Figure:
    """
    Plots a metric vs. time using Plotly, with a line showing the best-so-far metric value.

    Parameters:
        dataframe (pd.DataFrame): DataFrame containing the data.
        time_col (str): Column name for timestamps (default is 'created').
        metric_col (str): Column name for the metric being plotted (default is 'accuracy').

    Returns:
        plotly.graph_objects.Figure: The generated interactive Plotly figure.

    Example:
        >>> import pandas as pd
        >>> data = {
        ...     'created': [
        ...         '2025-02-08 09:45:00', '2025-02-08 09:45:30', '2025-02-08 09:46:00',
        ...         '2025-02-08 09:46:30', '2025-02-08 09:47:00', '2025-02-08 09:47:30',
        ...         '2025-02-08 09:48:00', '2025-02-08 09:48:30', '2025-02-08 09:49:00',
        ...         '2025-02-08 09:49:30'
        ...     ],
        ...     'accuracy': [0.942, 0.958, 0.966, 0.958, 0.966, 0.975, 0.966, 0.958, 0.975, 0.966]
        ... }
        >>> df = pd.DataFrame(data)
        >>> fig = plot_metric_vs_time(df)
    """
    df = dataframe.copy()
    df[time_col] = pd.to_datetime(df[time_col])

    # Calculate best-so-far metric value
    df.sort_values(by=time_col, inplace=True)
    df["best_so_far"] = df[metric_col].cummax()

    # Scatter plot of all points
    fig = px.scatter(
        df,
        x=time_col,
        y=metric_col,
        color=metric_col,
        title=f"{metric_col} v. {time_col}",
        labels={time_col: time_col.capitalize(), metric_col: metric_col.capitalize()},
        size=[10] * len(df),  # Fixed size for all points
    )

    # Add line for best-so-far
    fig.add_trace(
        go.Scatter(
            x=df[time_col],
            y=df["best_so_far"],
            mode="lines+markers",
            line={"color": "skyblue"},
            name="Best so far",
            showlegend=False,
        )
    )

    # Update layout
    fig.update_layout(
        xaxis_title=time_col.capitalize(),
        yaxis_title=metric_col.capitalize(),
        title={"x": 0.5},
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        height=400,
    )

    return fig


def plot_parameter_importance_and_correlation(results: dict, metric_name: str = "accuracy") -> go.Figure:
    """
    Plot parameter importance and correlation with respect to a metric using Plotly.
    Creates a 2x2 grid with separate plots for each score type.

    Args:
        results (dict): Output from calculate_feature_importance_and_correlation().
        metric_name (str): Name of the metric (e.g., "accuracy", "loss").

    Returns:
        plotly.graph_objects.Figure: Interactive 2x2 grid figure of bar plots.
    """
    # Convert the results dict to a DataFrame for easier plotting
    data = []
    for param, stats in results.items():
        data.append(
            {
                "Parameter": param,
                "Importance": stats["importance"],
                "Permutation Importance": stats["permutation_importance"],
                "Correlation (Pearson)": stats["pearson"],
                "Correlation (Spearman)": stats["spearman"],
            }
        )
    df = pd.DataFrame(data)
    df.sort_values("Importance", ascending=False, inplace=True)

    # Create a 2x2 subplot figure
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "Parameter Importance",
            "Pearson Correlation",
            "Spearman Correlation",
            "Permutation Importance",
        ],
    )

    # Define common properties
    param_order = df["Parameter"].tolist()

    # 1. Importance plot (top-left)
    fig.add_trace(
        go.Bar(x=df["Importance"], y=df["Parameter"], orientation="h", marker_color="royalblue"), row=1, col=1
    )

    # 2. Pearson correlation plot (top-right)
    fig.add_trace(
        go.Bar(
            x=df["Correlation (Pearson)"],
            y=df["Parameter"],
            orientation="h",
            marker_color=["seagreen" if v >= 0 else "crimson" for v in df["Correlation (Pearson)"]],
        ),
        row=1,
        col=2,
    )

    # 3. Spearman correlation plot (bottom-left)
    fig.add_trace(
        go.Bar(
            x=df["Correlation (Spearman)"],
            y=df["Parameter"],
            orientation="h",
            marker_color=["seagreen" if v >= 0 else "crimson" for v in df["Correlation (Spearman)"]],
        ),
        row=2,
        col=1,
    )

    # 4. Permutation importance plot (bottom-right)
    fig.add_trace(
        go.Bar(x=df["Permutation Importance"], y=df["Parameter"], orientation="h", marker_color="purple"), row=2, col=2
    )

    # Update layout
    fig.update_layout(
        title=f"Parameter Analysis for {metric_name}",
        height=max(600, 300 + 30 * len(df)),
        width=1000,
        showlegend=True,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
    )

    # Update axes
    fig.update_xaxes(title_text="Importance Score", row=1, col=1)
    fig.update_xaxes(title_text="Pearson Correlation", row=1, col=2)
    fig.update_xaxes(title_text="Spearman Correlation", row=2, col=1)
    fig.update_xaxes(title_text="Permutation Importance", row=2, col=2)

    # Ensure consistent y-axis ordering across all subplots
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_yaxes(categoryorder="array", categoryarray=param_order, row=i, col=j)
    fig.update_layout(showlegend=False)
    return fig


def plot_trial_timeline(
    df: pd.DataFrame,
    start_col: str = "start",
    end_col: str = "end",
    run_col: str = "run",
    status_col: str = "status",
    color_map: dict | None = None,
    title: str = "Timeline Plot",
) -> go.Figure:
    """
    Creates a Plotly timeline plot for trial runs.

    Parameters:
    - df: DataFrame containing trial data.
    - start_col: Name of the column containing start timestamps.
    - end_col: Name of the column containing end timestamps.
    - run_col: Name of the column identifying run IDs.
    - status_col: Name of the column specifying trial status.
    - color_map: Optional dict to specify colors for statuses.
    - title: Title of the plot.

    Example:
        >>> import pandas as pd
        >>> data = {
        ...     "start": ["2023-01-01 10:00:00", "2023-01-01 11:00:00", "2023-01-01 12:00:00"],
        ...     "end": ["2023-01-01 10:30:00", "2023-01-01 11:30:00", "2023-01-01 12:30:00"],
        ...     "run": ["Run 1", "Run 2", "Run 3"],
        ...     "status": ["finished", "failed", "pruned"]
        ... }
        >>> df = pd.DataFrame(data)
        >>> fig = plot_trial_timeline(df)

    """
    # Default color map if none is provided
    if color_map is None:
        color_map = {"finished": "blue", "failed": "red", "pruned": "orange"}

    # Ensure datetime columns are in proper format
    df[start_col] = pd.to_datetime(df[start_col])
    df[end_col] = pd.to_datetime(df[end_col])

    # Ensure timeline data is at least 1 second long
    zero_duration = df[start_col] == df[end_col]
    df.loc[zero_duration, end_col] += pd.Timedelta(seconds=1)

    # Create timeline plot
    fig = px.timeline(df, x_start=start_col, x_end=end_col, y=run_col, color=status_col, color_discrete_map=color_map)

    fig.update_layout(
        title=title, xaxis_title="Datetime", yaxis_title="Trial", yaxis_autorange="reversed", template="plotly_white"
    )

    return fig
