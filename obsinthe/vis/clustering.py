from obsinthe.deps import check_dependencies


check_dependencies("vis")

import numpy as np
from plotly import graph_objects as go

from obsinthe.alerts.grouping import AlertsClustering


def plot_clustering(ac: AlertsClustering):
    fig = go.Figure()

    unique_dbscan_labels = np.unique(ac.labels_)

    for c in unique_dbscan_labels:
        # index where points are of this grouping
        c_pts = ac.labels_ == c

        fig.add_trace(
            go.Scatter3d(
                x=ac.one_hot_trans_[c_pts, 0],
                y=ac.one_hot_trans_[c_pts, 1],
                z=ac.one_hot_trans_[c_pts, 2],
                name=f"grouping: {c}",
                mode="markers",
                marker=dict(size=2),
                text=ac.one_hot[c_pts].index,
            )
        )

    return fig
