import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def plot_metric_vs_time(dataframe, time_col="created", metric_col="accuracy"):
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


def plot_parameter_importance_and_correlation(results: dict, metric_name: str = "accuracy"):
    """
    Plot parameter importance and correlation with respect to a metric using Plotly.

    Args:
        results (dict): Output from calculate_feature_importance_and_correlation().
        metric_name (str): Name of the metric (e.g., "accuracy", "loss").

    Returns:
        plotly.graph_objects.Figure: Interactive bar plot figure.
    """
    # Convert the results dict to a DataFrame for easier plotting
    data = []
    for param, stats in results.items():
        data.append(
            {
                "Parameter": param,
                "Importance": stats["importance"],
                "Correlation (Pearson)": stats["pearson"],
            }
        )
    df = pd.DataFrame(data)
    df.sort_values("Importance", ascending=False, inplace=True)

    # Bar plot with grouped Importance and Correlation
    fig = go.Figure()

    # Importance bars (blue)
    fig.add_trace(
        go.Bar(x=df["Importance"], y=df["Parameter"], orientation="h", name="Importance", marker_color="royalblue")
    )

    # Correlation bars (green/red based on sign)
    fig.add_trace(
        go.Bar(
            x=df["Correlation (Pearson)"],
            y=df["Parameter"],
            orientation="h",
            name="Correlation (Pearson)",
            marker_color=["seagreen" if v >= 0 else "crimson" for v in df["Correlation (Pearson)"]],
            opacity=0.6,
        )
    )

    # Layout
    fig.update_layout(
        title=f"Parameter importance with respect to {metric_name}",
        barmode="overlay",
        xaxis_title="Score",
        yaxis_title="Config parameter",
        height=400 + 30 * len(df),
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        margin={"l": 120, "r": 20, "t": 60, "b": 40},
    )

    return fig
