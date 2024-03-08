import tempfile
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs

import pytest
import responses

from obsinthe.prometheus.data import (
    raw_to_instant_df,
    raw_to_range_df,
    raw_to_df,
    range_df_to_range_intervals_df,
    range_intervals_df_to_intervals_df,
    range_df_to_intervals_df,
    intervals_merge_overlaps,
    intervals_concat_days,
    group_by_time,
    one_hot_encode,
)
from obsinthe.testing.prometheus import (
    PromInstantDatasetBuilder,
    PromRangeDatasetBuilder,
)


def test_raw_to_instant_df(assert_df):
    df = raw_to_instant_df(instant_query_data())
    assert_df(
        df,
        """
   foo  timestamp  value
0  bar 2024-01-01   42.0
1  baz 2024-01-01    0.0
        """,
    )

    df = raw_to_instant_df(instant_query_data_extra_columns(), columns=["foo"])
    assert_df(
        df,
        """
   foo            extra  timestamp  value
0  bar   {'baz': 'qux'} 2024-01-01   42.0
1  baz  {'baz': 'quux'} 2024-01-01    0.0
        """,
    )


def test_raw_to_range_df(assert_df):
    df = raw_to_range_df(range_query_data())
    assert_df(
        df,
        """
   foo                                                        values
0  bar  [1704067320.0, 42.0, 1704067380.0, 42.0, 1704067440.0, 42.0]
1  baz     [1704067380.0, 3.0, 1704067440.0, 4.0, 1704067500.0, 5.0]
        """,
    )

    df = raw_to_range_df(range_query_data_extra_columns(), columns=["foo"])
    assert_df(
        df[["foo", "extra"]],
        """
   foo            extra
0  bar   {'baz': 'qux'}
1  baz  {'baz': 'quux'}
    """,
    )


def test_raw_to_df(assert_df):
    df = raw_to_df(instant_query_data())
    # detect the expected format (value vs. values) based on input data
    assert list(df.columns) == ["foo", "timestamp", "value"]

    df = raw_to_df(instant_query_data_extra_columns(), columns=["foo"])
    assert list(df.columns) == ["foo", "extra", "timestamp", "value"]

    df = raw_to_df(range_query_data())
    assert list(df.columns) == ["foo", "values"]

    df = raw_to_df(range_query_data_extra_columns(), columns=["foo"])
    assert list(df.columns) == ["foo", "extra", "values"]


def test_range_df_to_range_intervals_df(assert_df):
    range_df = raw_to_range_df(range_query_intervals_data())

    # merge overlaps and intervals within 60 seconds
    intervals_df = range_df_to_range_intervals_df(range_df, 60)
    assert_df(
        intervals_df,
        """
   foo                                                     intervals
0  bar                                [(1704067320.0, 1704067680.0)]
1  baz  [(1704067260.0, 1704067380.0), (1704067500.0, 1704067680.0)]
        """,
    )

    # try increasing the overlaps to 120 seconds
    intervals_df = range_df_to_range_intervals_df(range_df, 120)
    assert_df(
        intervals_df,
        """
   foo                       intervals
0  bar  [(1704067320.0, 1704067680.0)]
1  baz  [(1704067260.0, 1704067680.0)]
        """,
    )


def instant_query_data():
    builder = PromInstantDatasetBuilder()
    builder.ts({"foo": "bar"}).value(42)
    builder.ts({"foo": "baz"}).value(lambda t: t.minute)
    return builder.build_raw()


def instant_query_data_extra_columns():
    builder = PromInstantDatasetBuilder()
    builder.ts({"foo": "bar", "baz": "qux"}).value(42)
    builder.ts({"foo": "baz", "baz": "quux"}).value(lambda t: t.minute)
    return builder.build_raw()


def range_query_data():
    builder = PromRangeDatasetBuilder()
    builder.ts({"foo": "bar"}).interval(timedelta(minutes=2), timedelta(minutes=4), 42)
    builder.ts({"foo": "baz"}).interval(
        timedelta(minutes=3), timedelta(minutes=5), lambda t: t.minute
    )
    return builder.build_raw()


def range_query_intervals_data():
    builder = PromRangeDatasetBuilder()
    ts1 = builder.ts({"foo": "bar"})
    ts1.interval(timedelta(minutes=2), timedelta(minutes=4), 1)
    ts1.interval(timedelta(minutes=4), timedelta(minutes=6), 1)
    ts1.interval(timedelta(minutes=7), timedelta(minutes=8), 1)

    ts2 = builder.ts({"foo": "baz"})
    ts2.interval(timedelta(minutes=1), timedelta(minutes=3), 1)
    ts2.interval(timedelta(minutes=5), timedelta(minutes=7), 1)
    ts2.interval(timedelta(minutes=6), timedelta(minutes=8), 1)
    return builder.build_raw()


def range_query_data_extra_columns():
    builder = PromRangeDatasetBuilder()
    builder.ts({"foo": "bar", "baz": "qux"}).interval(
        timedelta(minutes=2), timedelta(minutes=4), 42
    )
    builder.ts({"foo": "baz", "baz": "quux"}).interval(
        timedelta(minutes=3), timedelta(minutes=5), lambda t: t.minute
    )
    return builder.build_raw()
