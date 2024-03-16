from datetime import timedelta

from obsinthe.alerts.grouping import alerts_clustering_dbscan
from obsinthe.alerts.grouping import alerts_groups_one_hot
from obsinthe.prometheus.data import RangeDataset
from obsinthe.testing.prometheus.alerts import AlertsDatasetBuilder
from obsinthe.testing.prometheus.builder import DEFAULT_END_TIME
from obsinthe.testing.prometheus.builder import DEFAULT_START_TIME
from obsinthe.vis.clustering import plot_clustering


def test_alerts_clustering_dbscan():
    # Create a dataset of alerts
    alerts_data = AlertsDatasetBuilder(DEFAULT_START_TIME, DEFAULT_END_TIME).build_raw()
    range_ds = RangeDataset.from_raw(alerts_data)
    # Create a dataset of intervals
    intervals_ds = range_ds.to_intervals_ds(timedelta(minutes=1))
    # Encode the groups
    one_hot = alerts_groups_one_hot(
        intervals_ds,
        groupby_columns=["instance_id"],
        group_tolerance=timedelta(minutes=30),
    )
    # Cluster the alerts
    ac = alerts_clustering_dbscan(
        one_hot,
        eps=1,
        n_neighbors=2,
        min_samples=2,
        min_dist=0.1,
    )

    # Check expected groups were properly identified.
    assert {
        "KubeDeploymentReplicasMismatch",
        "KubeNodeNotReady",
        "TargetDown",
    } in ac.groups_

    assert {
        "ElasticsearchClusterNotHealthy",
        "ElasticsearchJVMHeapUseHigh",
    } in ac.groups_

    # Just a basic test to check there are no exceptions.
    plot_clustering(ac)
