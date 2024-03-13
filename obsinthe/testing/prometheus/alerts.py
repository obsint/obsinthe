import random
from datetime import timedelta

from obsinthe.testing.prometheus.builder import PromRangeDatasetBuilder


def alerts_simulation_builder(start, end):
    start_rnd = random.Random(1)
    end_rnd = random.Random(1)
    flap_rnd = random.Random(1)
    builder = PromRangeDatasetBuilder(start, end)
    jitter = 5
    data = {
        "a": {"labels": {"alertname": "TargetDown", "severity": "warning"}},
        "b": {
            "labels": {
                "alertname": "KubeDeploymentReplicasMismatch",
                "severity": "warning",
            }
        },
        "c": {"labels": {"alertname": "KubeNodeNotReady", "severity": "critical"}},
        "d": {
            "labels": {
                "alertname": "ElasticsearchClusterNotHealthy",
                "severity": "warning",
            }
        },
        "e": {
            "labels": {"alertname": "ElasticsearchJVMHeapUseHigh", "severity": "info"},
            "flap": True,
        },
    }

    related_groups = [["a", "b", "c"], ["d", "e"]]

    for i, group in enumerate(related_groups):
        group_start = timedelta(minutes=30 * i)
        for g in group:
            alert_start = group_start + timedelta(
                minutes=start_rnd.randrange(-1 * jitter, jitter)
            )
            alert_end = group_start + timedelta(
                minutes=15 + end_rnd.randrange(-1 * jitter, jitter)
            )
            simulate_alerts_ts(
                data[g],
                {"cluster_id": str(i + 1)},
                alert_start,
                alert_end,
                builder,
                flap_rnd,
            )

    return builder


def simulate_alerts_ts(data, extra_labels, start, end, builder, rnd):
    labels = {**data["labels"], **extra_labels}
    if data.get("flap", False):
        active = True
        s = start
        intervals = []
        for m in range(int((end - start).total_seconds() / 60)):
            t = start + timedelta(minutes=m)
            new_active = rnd.choice([True, False])

            if active == new_active:
                continue

            if new_active:
                s = t
            else:
                intervals.append([s, t])

            active = new_active

    else:
        intervals = [[start, end]]

    for start, end in intervals:
        builder.ts(labels).interval(start, end, 1)
