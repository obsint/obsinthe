from datetime import timedelta

from obsinthe.prometheus.data import DatasetCollection
from obsinthe.prometheus.data import group_by_time
from obsinthe.prometheus.data import InstantDataset
from obsinthe.prometheus.data import intervals_concat_days
from obsinthe.prometheus.data import IntervalsDataset
from obsinthe.prometheus.data import one_hot_encode
from obsinthe.prometheus.data import RangeDataset
from obsinthe.prometheus.data import raw_to_ds
from obsinthe.testing.prometheus.builder import DEFAULT_START_TIME
from obsinthe.testing.prometheus.builder import PromInstantDatasetBuilder
from obsinthe.testing.prometheus.builder import PromRangeDatasetBuilder


def test_instant_dataset_from_raw(assert_df):
    ds = InstantDataset.from_raw(instant_query_data())
    assert_df(
        ds.df,
        """
   foo  timestamp  value
0  bar 2024-01-01   42.0
1  baz 2024-01-01    0.0
        """,
    )

    ds = InstantDataset.from_raw(instant_query_data(extra_labels=True), columns=["foo"])
    assert_df(
        ds.df,
        """
   foo            extra  timestamp  value
0  bar   {'baz': 'qux'} 2024-01-01   42.0
1  baz  {'baz': 'quux'} 2024-01-01    0.0
        """,
    )


def test_range_dataset_from_raw(assert_df):
    ds = RangeDataset.from_raw(range_query_data())
    assert_df(
        ds.df,
        """
   foo                                                        values
0  bar  [1704067320.0, 42.0, 1704067380.0, 42.0, 1704067440.0, 42.0]
1  baz     [1704067380.0, 3.0, 1704067440.0, 4.0, 1704067500.0, 5.0]
        """,
    )

    ds = RangeDataset.from_raw(range_query_data(extra_labels=True), columns=["foo"])
    assert_df(
        ds.df[["foo", "extra"]],
        """
   foo            extra
0  bar   {'baz': 'qux'}
1  baz  {'baz': 'quux'}
    """,
    )


def test_raw_to_ds():
    ds = raw_to_ds(instant_query_data())
    # detect the expected format (value vs. values) based on input data
    assert list(ds.df.columns) == ["foo", "timestamp", "value"]

    ds = raw_to_ds(instant_query_data(extra_labels=True), columns=["foo"])
    assert list(ds.df.columns) == ["foo", "extra", "timestamp", "value"]

    ds = raw_to_ds(range_query_data())
    assert list(ds.df.columns) == ["foo", "values"]

    ds = raw_to_ds(range_query_data(extra_labels=True), columns=["foo"])
    assert list(ds.df.columns) == ["foo", "extra", "values"]


def test_range_dataset_to_range_intervals_ds(assert_df):
    range_ds = RangeDataset.from_raw(range_query_intervals_data())

    # merge overlaps and intervals within 60 seconds
    range_intervals_ds = range_ds.to_range_intervals_ds(timedelta(minutes=1))
    assert_df(
        range_intervals_ds.df,
        """
   foo                                                     intervals
0  bar                                [(1704067320.0, 1704067680.0)]
1  baz  [(1704067260.0, 1704067380.0), (1704067500.0, 1704067680.0)]
        """,
    )

    # try increasing the overlaps to 2 minutes
    range_intervals_ds = range_ds.to_range_intervals_ds(timedelta(minutes=2))
    assert_df(
        range_intervals_ds.df,
        """
   foo                       intervals
0  bar  [(1704067320.0, 1704067680.0)]
1  baz  [(1704067260.0, 1704067680.0)]
        """,
    )


def test_range_intervals_dataset_to_intervals_ds(assert_df):
    range_ds = RangeDataset.from_raw(range_query_intervals_data())
    range_intervals_ds = range_ds.to_range_intervals_ds(timedelta(minutes=1))
    intervals_ds = range_intervals_ds.to_intervals_ds()

    # expands the intervals to individual rows and converts timestamps to datetime
    assert_df(
        intervals_ds.df,
        """
   foo                     start                       end
0  bar 2024-01-01 00:02:00+00:00 2024-01-01 00:08:00+00:00
1  baz 2024-01-01 00:01:00+00:00 2024-01-01 00:03:00+00:00
2  baz 2024-01-01 00:05:00+00:00 2024-01-01 00:08:00+00:00
        """,
    )


def test_range_dataset_to_intervals_ds(assert_df):
    range_ds = RangeDataset.from_raw(range_query_intervals_data())
    intervals_ds = range_ds.to_intervals_ds(timedelta(minutes=1))

    # does conversion to range intervals and then to intervals
    assert_df(
        intervals_ds.df,
        """
   foo                     start                       end
0  bar 2024-01-01 00:02:00+00:00 2024-01-01 00:08:00+00:00
1  baz 2024-01-01 00:01:00+00:00 2024-01-01 00:03:00+00:00
2  baz 2024-01-01 00:05:00+00:00 2024-01-01 00:08:00+00:00
        """,
    )


def test_intervals_merge_overlaps(assert_df):
    def sub_intervals_str(sub_intervals):
        return "\n".join(
            [str([start.isoformat(), end.isoformat()]) for start, end in sub_intervals]
        )

    range_ds = RangeDataset.from_raw(range_query_intervals_data())
    intervals_ds = range_ds.to_intervals_ds(timedelta(minutes=1))

    # merges overlapping intervals
    intervals_ds = intervals_ds.merge_overlaps(timedelta(minutes=2))
    assert_df(
        intervals_ds.df[["foo", "start", "end"]],
        """
   foo                     start                       end
0  bar 2024-01-01 00:02:00+00:00 2024-01-01 00:08:00+00:00
1  baz 2024-01-01 00:01:00+00:00 2024-01-01 00:08:00+00:00
        """,
    )

    # also includes sub-intervals
    assert (
        sub_intervals_str(intervals_ds.df["sub_intervals"].iloc[0])
        == """['2024-01-01T00:02:00+00:00', '2024-01-01T00:08:00+00:00']"""
    )

    assert (
        sub_intervals_str(intervals_ds.df["sub_intervals"].iloc[1])
        == """
['2024-01-01T00:01:00+00:00', '2024-01-01T00:03:00+00:00']
['2024-01-01T00:05:00+00:00', '2024-01-01T00:08:00+00:00']
    """.strip()
    )

    # test multi-columns scenario a merge on a specific column
    range_ds = RangeDataset.from_raw(range_query_intervals_data(extra_labels=True))
    intervals_ds = range_ds.to_intervals_ds(timedelta(minutes=1))

    intervals_ds = intervals_ds.merge_overlaps(timedelta(minutes=2), columns=["baz"])
    assert_df(
        intervals_ds.df[["baz", "start", "end"]],
        """
   baz                     start                       end
0  qux 2024-01-01 00:01:00+00:00 2024-01-01 00:08:00+00:00
    """,
    )

    assert (
        sub_intervals_str(intervals_ds.df["sub_intervals"].iloc[0])
        == """
['2024-01-01T00:01:00+00:00', '2024-01-01T00:03:00+00:00']
['2024-01-01T00:02:00+00:00', '2024-01-01T00:08:00+00:00']
['2024-01-01T00:05:00+00:00', '2024-01-01T00:08:00+00:00']
""".strip()
    )

    empty_ds = intervals_ds.fmap(lambda df: df.iloc[0:0])
    assert empty_ds.merge_overlaps(timedelta(minutes=2)).df.empty


def test_intervals_concat_days(assert_df):
    intervals_ds_col = multi_day_intervals_ds_collection()

    concat_ds = intervals_concat_days(intervals_ds_col)

    # By default concat only intervals that match exactly.
    assert_df(
        concat_ds.df.sort_values("start"),
        """
  foo                     start                       end
4   a 2024-01-01 00:00:00+00:00 2024-01-04 00:00:00+00:00
0   b 2024-01-01 21:00:00+00:00 2024-01-01 22:15:00+00:00
1   b 2024-01-01 22:20:00+00:00 2024-01-01 23:56:00+00:00
2   b 2024-01-02 00:04:00+00:00 2024-01-02 02:00:00+00:00
3   b 2024-01-02 02:04:00+00:00 2024-01-02 03:00:00+00:00
    """,
    )

    # Allow for additional tolerance.
    concat_ds = intervals_concat_days(intervals_ds_col, timedelta(minutes=10))

    # It touches only the day boundaries: the intervals inside the same day are
    # not merged even if they are close to each other.
    assert_df(
        concat_ds.df.sort_values("start"),
        """
  foo                     start                       end
3   a 2024-01-01 00:00:00+00:00 2024-01-04 00:00:00+00:00
0   b 2024-01-01 21:00:00+00:00 2024-01-01 22:15:00+00:00
2   b 2024-01-01 22:20:00+00:00 2024-01-02 02:00:00+00:00
1   b 2024-01-02 02:04:00+00:00 2024-01-02 03:00:00+00:00
    """,
    )


def test_group_by_time(assert_df):
    intervals_ds = alerts_bursts_intervals_ds()

    # Group with no tolerance.
    grouped_df = group_by_time(intervals_ds.df, "start")
    assert_df(
        grouped_df,
        """
  alert                     start                       end  group_id
0     a 2024-01-01 00:03:00+00:00 2024-01-01 00:10:00+00:00         0
1     b 2024-01-01 00:03:00+00:00 2024-01-01 00:10:00+00:00         0
2     c 2024-01-01 00:06:00+00:00 2024-01-01 00:13:00+00:00         1
3     d 2024-01-01 00:09:00+00:00 2024-01-01 00:14:00+00:00         2
4     e 2024-01-01 00:13:00+00:00 2024-01-01 00:20:00+00:00         3
    """,
    )

    # Group with 3 minutes tolerance.
    grouped_df = group_by_time(intervals_ds.df, "start", tolerance=timedelta(minutes=3))
    assert_df(
        grouped_df,
        """
  alert                     start                       end  group_id
0     a 2024-01-01 00:03:00+00:00 2024-01-01 00:10:00+00:00         0
1     b 2024-01-01 00:03:00+00:00 2024-01-01 00:10:00+00:00         0
2     c 2024-01-01 00:06:00+00:00 2024-01-01 00:13:00+00:00         0
3     d 2024-01-01 00:09:00+00:00 2024-01-01 00:14:00+00:00         0
4     e 2024-01-01 00:13:00+00:00 2024-01-01 00:20:00+00:00         1
    """,
    )

    # Group with explicit group_id columns name.
    grouped_df = group_by_time(intervals_ds.df, "start", group_id_column="group_column")
    assert list(grouped_df.columns) == ["alert", "start", "end", "group_column"]

    # Grouped with extra columns.
    intervals_ds = alerts_bursts_intervals_ds(extra_labels=True)
    grouped_df = group_by_time(
        intervals_ds.df,
        "start",
        extra_groupby_columns=["instance"],
        tolerance=timedelta(minutes=3),
    )
    assert_df(
        grouped_df,
        """
  alert instance                     start                       end group_id
0     a        1 2024-01-01 00:03:00+00:00 2024-01-01 00:10:00+00:00      1-0
1     b        1 2024-01-01 00:03:00+00:00 2024-01-01 00:10:00+00:00      1-0
2     c        2 2024-01-01 00:06:00+00:00 2024-01-01 00:13:00+00:00      2-0
3     d        2 2024-01-01 00:09:00+00:00 2024-01-01 00:14:00+00:00      2-0
4     e        2 2024-01-01 00:13:00+00:00 2024-01-01 00:20:00+00:00      2-1
    """,
    )


def test_one_hot_encode(assert_df):
    intervals_ds = alerts_bursts_intervals_ds(extra_labels=True)
    grouped_df = group_by_time(
        intervals_ds.df,
        "start",
        extra_groupby_columns=["instance"],
        tolerance=timedelta(minutes=3),
    )

    one_hot_df = one_hot_encode(grouped_df, "group_id", "alert")
    assert_df(
        one_hot_df,
        """
alert     a  b  c  d  e
group_id
1-0       1  1  0  0  0
2-0       0  0  1  1  0
2-1       0  0  0  0  1
    """,
    )


def instant_query_data(extra_labels=False):
    builder = PromInstantDatasetBuilder()

    labels = {"foo": "bar"}
    if extra_labels:
        labels["baz"] = "qux"
    builder.ts(labels).value(42)

    labels = {"foo": "baz"}
    if extra_labels:
        labels["baz"] = "quux"
    builder.ts(labels).value(lambda t: t.minute)
    return builder.build_raw()


def range_query_data(extra_labels=False):
    builder = PromRangeDatasetBuilder()

    labels = {"foo": "bar"}
    if extra_labels:
        labels["baz"] = "qux"
    builder.ts(labels).interval(timedelta(minutes=2), timedelta(minutes=4), 42)

    labels = {"foo": "baz"}
    if extra_labels:
        labels["baz"] = "quux"
    builder.ts(labels).interval(
        timedelta(minutes=3), timedelta(minutes=5), lambda t: t.minute
    )
    return builder.build_raw()


def range_query_intervals_data(extra_labels=False):
    builder = PromRangeDatasetBuilder()

    labels = {"foo": "bar"}
    if extra_labels:
        labels["baz"] = "qux"
    ts1 = builder.ts(labels)
    ts1.interval(timedelta(minutes=2), timedelta(minutes=4), 1)
    ts1.interval(timedelta(minutes=4), timedelta(minutes=6), 1)
    ts1.interval(timedelta(minutes=7), timedelta(minutes=8), 1)

    labels = {"foo": "baz"}
    if extra_labels:
        labels["baz"] = "qux"
    ts2 = builder.ts(labels)
    ts2.interval(timedelta(minutes=1), timedelta(minutes=3), 1)
    ts2.interval(timedelta(minutes=5), timedelta(minutes=7), 1)
    ts2.interval(timedelta(minutes=6), timedelta(minutes=8), 1)
    return builder.build_raw()


def multi_day_intervals_ds():
    builder = PromRangeDatasetBuilder(
        start=DEFAULT_START_TIME, end=DEFAULT_START_TIME + timedelta(days=4)
    )
    ts_a_1 = builder.ts({"foo": "a", "day": "1"})
    ts_a_1.interval(timedelta(minutes=0), timedelta(hours=24), 1)
    ts_b_1 = builder.ts({"foo": "b", "day": "1"})
    ts_b_1.interval(timedelta(hours=21), timedelta(hours=22, minutes=15), 1)
    ts_b_1.interval(timedelta(hours=22, minutes=20), timedelta(hours=23, minutes=56), 1)

    ts_a_2 = builder.ts({"foo": "a", "day": "2"})
    ts_a_2.interval(timedelta(days=1), timedelta(days=2), 1)
    ts_b_2 = builder.ts({"foo": "b", "day": "2"})
    ts_b_2.interval(timedelta(days=1, minutes=4), timedelta(days=1, hours=2), 1)
    ts_b_2.interval(
        timedelta(days=1, hours=2, minutes=4), timedelta(days=1, hours=3), 1
    )

    ts_a_3 = builder.ts({"foo": "a", "day": "3"})
    ts_a_3.interval(timedelta(days=2), timedelta(days=3), 1)

    data = builder.build_raw()
    range_ds = RangeDataset.from_raw(data)
    return range_ds.to_intervals_ds(timedelta(minutes=1))


def multi_day_intervals_ds_collection():
    intervals_ds = multi_day_intervals_ds()

    # Split the intervals by day to simulate a multi-day scenario.
    datasets = []
    for _, day_df in intervals_ds.df.groupby("day"):
        datasets.append(IntervalsDataset(day_df.drop(columns=["day"])))

    return DatasetCollection(datasets)


def alerts_bursts_intervals_ds(extra_labels=False):
    builder = PromRangeDatasetBuilder()

    labels = {
        "a": {"alert": "a", "instance": "1"},
        "b": {"alert": "b", "instance": "1"},
        "c": {"alert": "c", "instance": "2"},
        "d": {"alert": "d", "instance": "2"},
        "e": {"alert": "e", "instance": "2"},
    }

    if not extra_labels:

        def clean_labels(labels):
            return {k: v for k, v in labels.items() if k == "alert"}

        labels = {k: clean_labels(v) for k, v in labels.items()}

    # Alert a starting at the same time as alert b.
    # Alert c starting 3 minutes later than a.
    # Alert d starting 3 minutes later than c.
    # Alert e starting 4 minutes later than d.
    builder.ts(labels["a"]).interval(timedelta(minutes=3), timedelta(minutes=10), 1)
    builder.ts(labels["b"]).interval(timedelta(minutes=3), timedelta(minutes=10), 1)
    builder.ts(labels["c"]).interval(timedelta(minutes=6), timedelta(minutes=13), 1)
    builder.ts(labels["d"]).interval(timedelta(minutes=9), timedelta(minutes=14), 1)
    builder.ts(labels["e"]).interval(timedelta(minutes=13), timedelta(minutes=20), 1)

    data = builder.build_raw()
    range_ds = RangeDataset.from_raw(data)
    return range_ds.to_intervals_ds(timedelta(minutes=60))
