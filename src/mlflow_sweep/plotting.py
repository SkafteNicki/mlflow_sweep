import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


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
            line=dict(color="skyblue"),
            name="Best so far",
            showlegend=False,
        )
    )

    # Update layout
    fig.update_layout(
        xaxis_title=time_col.capitalize(),
        yaxis_title=metric_col.capitalize(),
        title=dict(x=0.5),
        margin=dict(l=40, r=20, t=40, b=40),
        height=400,
    )

    return fig
