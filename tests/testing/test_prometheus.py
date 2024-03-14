from datetime import datetime
from datetime import timedelta

from obsinthe.prometheus.data import RangeDS
from obsinthe.testing.prometheus import PromInstantDatasetBuilder
from obsinthe.testing.prometheus import PromRangeDatasetBuilder
from obsinthe.testing.prometheus import TimeSeriesBuilder
from obsinthe.testing.prometheus.alerts import AlertsDataSetBuilder


def test_time_series_builder():
    ts1 = TimeSeriesBuilder({"foo": "bar"})
    ts1.sample(timedelta(minutes=2), 42)
    ts1.interval(timedelta(minutes=4), timedelta(minutes=6), lambda t: t.minute * 2)

    ts1_data = ts1.build(
        datetime(2021, 1, 1), datetime(2021, 1, 2), timedelta(minutes=1)
    )
    assert ts1_data == [
        (1609459320, 42.0),
        (1609459440, 8.0),
        (1609459500, 10.0),
        (1609459560, 12.0),
    ]

    ts2 = TimeSeriesBuilder({"foo": "baz"})
    ts2.value(lambda t: t.minute * 3)
    ts2_data = ts2.build(
        datetime(2021, 1, 1), datetime(2021, 1, 2), timedelta(minutes=20)
    )
    assert len(ts2_data) == 73
    assert ts2_data[:5] == [
        (1609459200, 0.0),
        (1609460400, 60.0),
        (1609461600, 120.0),
        (1609462800, 0.0),
        (1609464000, 60.0),
    ]


def test_prom_range_dataset_builder():
    builder = PromRangeDatasetBuilder()
    builder.ts({"foo": "bar"}).interval(timedelta(minutes=2), timedelta(minutes=4), 42)
    builder.ts({"foo": "baz"}).interval(
        timedelta(minutes=3), timedelta(minutes=5), lambda t: t.minute
    )

    data = builder.build_raw()

    assert data == [
        {
            "metric": {"foo": "bar"},
            "values": [
                [1704067320, "42.0"],
                [1704067380, "42.0"],
                [1704067440, "42.0"],
            ],
        },
        {
            "metric": {"foo": "baz"},
            "values": [[1704067380, "3.0"], [1704067440, "4.0"], [1704067500, "5.0"]],
        },
    ]


def test_prom_instant_dataset_builder():
    builder = PromInstantDatasetBuilder(datetime(2024, 1, 1, 14, 30))
    builder.ts({"foo": "bar"}).value(42)
    builder.ts({"foo": "baz"}).value(lambda t: t.minute)
    data = builder.build_raw()

    assert data == [
        {"metric": {"foo": "bar"}, "value": [1704119400, "42.0"]},
        {"metric": {"foo": "baz"}, "value": [1704119400, "30.0"]},
    ]


def test_alerts_dataset_builder():
    builder = AlertsDataSetBuilder(
        datetime(2024, 1, 1, 14, 30), datetime(2024, 1, 1, 15, 30)
    )

    data = builder.build_raw()
    range_ds = RangeDS.from_raw(data)

    # Just a simple check that the data was generated without errors.
    assert "TargetDown" in set(range_ds.df["alertname"])
    assert range_ds.df["alertname"].nunique() == 28
