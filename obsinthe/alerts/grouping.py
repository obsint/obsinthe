from obsinthe.deps import check_dependencies


check_dependencies("ml")

import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from umap import UMAP

from obsinthe.prometheus.data import group_by_time
from obsinthe.prometheus.data import IntervalsDataset
from obsinthe.prometheus.data import one_hot_encode


class AlertsClustering:
    one_hot = None
    one_hot_trans_ = None
    labels_ = None


class AlertsClusteringDBSCAN(AlertsClustering):
    def __init__(self, eps=0.5, min_samples=5, n_neighbors=5, min_dist=0):
        self.eps = eps
        self.min_samples = min_samples
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist

    def fit(self, one_hot):
        self.one_hot = one_hot
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*inverse_transform will be unavailable.*"
            )
            warnings.filterwarnings("ignore", message=".*Use no seed for parallelism.*")
            self.fit_transform()

        self.fit_dbscan()

    def fit_transform(self):
        # reducing to 3 dimensions
        self.umap_ = UMAP(
            n_components=3,
            metric="hamming",
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            random_state=42,
        )

        self.umap_.fit(self.one_hot)
        self.one_hot_trans_ = self.umap_.transform(self.one_hot)

    def fit_dbscan(self):
        self.dbscan_ = DBSCAN(
            eps=self.eps,
            # a pair is good enough, ceiling could be 10 (post-processing)
            min_samples=self.min_samples,
            metric="euclidean",
            n_jobs=-1,
        )
        self.dbscan_.fit(self.one_hot_trans_)
        self.labels_ = self.dbscan_.labels_

        self.groups_ = []
        for g in np.unique(self.labels_):
            if g == -1:
                continue

            group = set(self.one_hot.index[self.labels_ == g])
            self.groups_.append(group)


def alerts_groups_one_hot(
    ds: IntervalsDataset,
    group_tolerance=timedelta(minutes=5),
    groupby_columns=None,
    alert_id_column="alertname",
):
    groups_df = group_by_time(
        ds.df,
        "start",
        extra_groupby_columns=groupby_columns,
        tolerance=group_tolerance,
    )
    return one_hot_encode(groups_df, "group_id", alert_id_column)


def alerts_clustering_dbscan(one_hot: pd.DataFrame, **kwargs):
    one_hot_t = one_hot.T
    ret = AlertsClusteringDBSCAN(**kwargs)
    ret.fit(one_hot_t)
    return ret
