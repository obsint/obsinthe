try:
    import plotly.express as px
except ImportError as missing_imports:
    raise ImportError(
        """\
vis module requires extra dependencies which can be installed doing

$ pip install "obsinthe[vis]"

or

$ poetry install --extras vis
  """
    ) from missing_imports

from obsinthe.prometheus.data import IntervalsDS


def plot_alerts_timeline(intervals_ds: IntervalsDS):
    """Plot a timeline of alerts as a Gantt chart."""
    alerts = intervals_ds.df
    alerts = alerts.sort_values(["severity", "alertname"], ascending=[True, True])
    alerts_ = alerts.copy()
    alerts_["severity"] = alerts_["severity"].astype(str)

    fig = px.timeline(
        alerts_,
        x_start="start",
        x_end="end",
        y="alertname",
        color="severity",
        title="Timeline",
        color_discrete_map={
            "info": "blue",
            "warning": "orange",
            "critical": "red",
        },
    )
    fig.update_layout(showlegend=False)
    fig.update_traces(width=0.8)
    fig.update_layout(bargap=0.1, height=300)

    return fig
