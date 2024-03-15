import random
from datetime import timedelta

from obsinthe.testing.prometheus.builder import PromRangeDatasetBuilder

DATA = {
    "alerts": {
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
    },
    "groups": [
        {"alerts": ["a", "b", "c"], "frequency": 0.5},
        {"alerts": ["d", "e"], "frequency": 0.3},
    ],
}


class AlertsDatasetBuilder(PromRangeDatasetBuilder):
    def __init__(self, start, end, data=DATA, n_instances=100):
        super().__init__(start, end)
        self.alerts_data = data["alerts"]
        self.related_groups = data["groups"]
        self.n_instances = n_instances

        self.noise_alerts = [f"noise_{chr(c)}" for c in range(ord("a"), ord("z") + 1)]

        self.instance_ids = [str(i) for i in range(1, n_instances + 1)]

        self.alerts_default_duration = timedelta(minutes=30)

        self.rnd = random.Random(42)
        self.jitter = 5

        self.initialize()

    def initialize(self):
        for group in self.related_groups:
            self.add_group(group)

    def add_group(self, group):
        # Select instances to be affected by the group.
        frequency = group["frequency"]
        sample_instance_ids = self.rnd.sample(
            self.instance_ids, int(frequency * self.n_instances)
        )

        for instance_id in sample_instance_ids:
            group_start = self.start + self.rnd.random() * (self.end - self.start)
            for g in group["alerts"] + self.rnd.sample(self.noise_alerts, 1):
                if g in self.noise_alerts:
                    data = {"labels": {"alertname": g, "severity": "warning"}}
                else:
                    data = self.alerts_data[g]

                alert_start = group_start + timedelta(
                    minutes=self.rnd.randrange(-1 * self.jitter, self.jitter)
                )
                alert_duration = self.alerts_default_duration
                alert_duration += timedelta(
                    minutes=self.rnd.randrange(-1 * self.jitter, self.jitter)
                )
                alert_end = group_start + alert_duration
                self.simulate_alerts_ts(
                    labels={**data["labels"], "instance_id": instance_id},
                    flap=data.get("flap", False),
                    start=alert_start,
                    end=alert_end,
                )

    def simulate_alerts_ts(self, labels, flap, start, end):
        if flap:
            active = True
            s = start
            intervals = []
            for m in range(int((end - start).total_seconds() / 60)):
                t = start + timedelta(minutes=m)
                new_active = self.rnd.choice([True, False])

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
            self.ts(labels).interval(start, end, 1)
