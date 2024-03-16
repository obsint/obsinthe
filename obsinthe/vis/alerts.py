from obsinthe.deps import check_dependencies


check_dependencies("vis")

from typing import Callable
from typing import Optional
from typing import Union

import plotly.express as px

from obsinthe.prometheus.data import IntervalsDataset


def plot_alerts_timeline(
    intervals_ds: IntervalsDataset,
    alert_id: Optional[Union[str, Callable]] = None,
    height: int = 600,
):
    """Plot a timeline of alerts as a Gantt chart."""
    alerts = intervals_ds.df.copy()
    alert_id = alert_id or "alertname"
    if isinstance(alert_id, str):
        alerts["alert_id"] = alerts[alert_id]
    else:
        alerts["alert_id"] = alerts.apply(alert_id, axis=1)

    alerts.sort_values(["severity", "alert_id"], ascending=[True, True], inplace=True)
    alerts["severity"] = alerts["severity"].astype(str)

    fig = px.timeline(
        alerts,
        x_start="start",
        x_end="end",
        y="alert_id",
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
    fig.update_layout(bargap=0.1, height=height)

    return fig
