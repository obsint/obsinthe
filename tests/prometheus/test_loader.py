import tempfile
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse, parse_qs

import pytest
import responses

from obsinthe.prometheus.client import Client
from obsinthe.prometheus.loader import Loader
from obsinthe.testing.prometheus import (
    PromInstantDatasetBuilder,
    PromRangeDatasetBuilder,
)


EXAMPLE_URL = "http://prometheus.example.com"

TEST_TIME = datetime(2024, 1, 1, 12, tzinfo=timezone.utc)


@pytest.fixture
def loader():
    client = Client(f"{EXAMPLE_URL}", "token")
    return Loader(client)


@responses.activate()
def test_instant_query(loader, assert_df):
    mock_instant_query()

    df = loader.query("my_metric", TEST_TIME)

    assert_df(
        df,
        """
   foo           timestamp  value
0  bar 2024-01-01 12:00:00   42.0
1  baz 2024-01-01 12:00:00    0.0
        """,
    )


@responses.activate()
def test_range_query(loader, assert_df):
    mock_range_query()

    df = loader.query("my_metric", TEST_TIME)

    assert_df(
        df,
        """
   foo                                                        values
0  bar  [1704106920.0, 42.0, 1704106980.0, 42.0, 1704107040.0, 42.0]
1  baz     [1704106980.0, 3.0, 1704107040.0, 4.0, 1704107100.0, 5.0]
        """,
    )


@responses.activate()
def test_batch_query(loader, assert_df):
    mock_range_query()

    res = loader.batch_query(
        {
            TEST_TIME: ["foo", "bar", "baz"],
            TEST_TIME - timedelta(hours=1): ["baz", "qux", "quux"],
        },
        "my_metric{%s}",
        batch_size=2,
    )

    queries = requests_queries()

    # This also tests the order of the queries based on keys and order of items
    # in the input: useful for consistent repeatable key ordering in the cache.
    assert queries == [
        {"query": ["my_metric{baz|quux}"], "time": ["1704106800"]},
        {"query": ["my_metric{qux}"], "time": ["1704106800"]},
        {"query": ["my_metric{bar|baz}"], "time": ["1704110400"]},
        {"query": ["my_metric{foo}"], "time": ["1704110400"]},
    ]

    assert_df(
        res[0],
        """
   foo                                                        values  \\
0  bar  [1704106920.0, 42.0, 1704106980.0, 42.0, 1704107040.0, 42.0]
1  baz     [1704106980.0, 3.0, 1704107040.0, 4.0, 1704107100.0, 5.0]
2  bar  [1704106920.0, 42.0, 1704106980.0, 42.0, 1704107040.0, 42.0]
3  baz     [1704106980.0, 3.0, 1704107040.0, 4.0, 1704107100.0, 5.0]

                  timestamp
0 2024-01-01 11:00:00+00:00
1 2024-01-01 11:00:00+00:00
2 2024-01-01 11:00:00+00:00
3 2024-01-01 11:00:00+00:00
    """,
    )

    assert_df(
        res[1],
        """
   foo                                                        values  \\
0  bar  [1704106920.0, 42.0, 1704106980.0, 42.0, 1704107040.0, 42.0]
1  baz     [1704106980.0, 3.0, 1704107040.0, 4.0, 1704107100.0, 5.0]
2  bar  [1704106920.0, 42.0, 1704106980.0, 42.0, 1704107040.0, 42.0]
3  baz     [1704106980.0, 3.0, 1704107040.0, 4.0, 1704107100.0, 5.0]

                  timestamp
0 2024-01-01 12:00:00+00:00
1 2024-01-01 12:00:00+00:00
2 2024-01-01 12:00:00+00:00
3 2024-01-01 12:00:00+00:00
    """,
    )


@responses.activate()
def test_interval_query(loader, assert_df):
    mock_range_query()

    res = loader.interval_query(
        "my_metric{%s}", start=TEST_TIME - timedelta(days=2), end=TEST_TIME
    )

    queries = requests_queries()

    # This also tests the order of the queries based on keys and order of items
    # in the input: useful for consistent repeatable key ordering in the cache.
    assert queries == [
        {"query": ["my_metric{%s}"], "time": ["1703980800"]},  # 2023-12-31 0:0
        {"query": ["my_metric{%s}"], "time": ["1704067200"]},  # 2023-01-01 0:0
        {"query": ["my_metric{%s}"], "time": ["1704153600"]},  # 2023-01-02 0:0
    ]

    assert_df(
        res[0],
        """
   foo                                                        values
0  bar  [1704106920.0, 42.0, 1704106980.0, 42.0, 1704107040.0, 42.0]
1  baz     [1704106980.0, 3.0, 1704107040.0, 4.0, 1704107100.0, 5.0]
        """,
    )


@responses.activate()
def test_cache(assert_df):
    mock_range_query()
    cache_dir = tempfile.mkdtemp()

    client = Client(f"{EXAMPLE_URL}", "token")
    loader = Loader(client, cache_dir=cache_dir)

    res = loader.batch_query(
        {
            TEST_TIME: ["foo", "bar", "baz"],
            TEST_TIME - timedelta(hours=1): ["baz", "qux", "quux"],
        },
        "my_metric{%s}",
        batch_size=2,
        cache_key="foo",
    )

    assert len(responses.calls) == 4
    assert_df(
        res[0],
        """
   foo                                                        values  \\
0  bar  [1704106920.0, 42.0, 1704106980.0, 42.0, 1704107040.0, 42.0]
1  baz     [1704106980.0, 3.0, 1704107040.0, 4.0, 1704107100.0, 5.0]
2  bar  [1704106920.0, 42.0, 1704106980.0, 42.0, 1704107040.0, 42.0]
3  baz     [1704106980.0, 3.0, 1704107040.0, 4.0, 1704107100.0, 5.0]

                  timestamp
0 2024-01-01 11:00:00+00:00
1 2024-01-01 11:00:00+00:00
2 2024-01-01 11:00:00+00:00
3 2024-01-01 11:00:00+00:00
    """,
    )

    responses.reset()
    mock_range_query()
    res = loader.batch_query(
        {
            TEST_TIME: ["foo", "bar", "baz"],
            TEST_TIME - timedelta(hours=1): ["baz", "qux", "quux"],
        },
        "my_metric{%s}",
        batch_size=2,
        cache_key="foo",
    )

    # Same key => no new queries
    assert len(responses.calls) == 0
    assert_df(
        res[0],
        """
   foo                                                        values  \\
0  bar  [1704106920.0, 42.0, 1704106980.0, 42.0, 1704107040.0, 42.0]
1  baz     [1704106980.0, 3.0, 1704107040.0, 4.0, 1704107100.0, 5.0]
2  bar  [1704106920.0, 42.0, 1704106980.0, 42.0, 1704107040.0, 42.0]
3  baz     [1704106980.0, 3.0, 1704107040.0, 4.0, 1704107100.0, 5.0]

                  timestamp
0 2024-01-01 11:00:00+00:00
1 2024-01-01 11:00:00+00:00
2 2024-01-01 11:00:00+00:00
3 2024-01-01 11:00:00+00:00
    """,
    )

    responses.reset()
    mock_range_query()
    res = loader.batch_query(
        {
            TEST_TIME: ["foo", "bar", "baz"],
            TEST_TIME - timedelta(hours=1): ["baz", "qux", "quux"],
        },
        "my_metric{%s}",
        batch_size=2,
        cache_key="bar",
    )

    # Different key => new queries
    assert len(responses.calls) == 4

    responses.reset()
    mock_range_query()
    res = loader.batch_query(
        {
            TEST_TIME: ["qux", "bar", "baz"],
            TEST_TIME - timedelta(hours=1): ["baz", "qux", "quux"],
        },
        "my_metric{%s}",
        batch_size=2,
        cache_key="foo",
    )

    # Different sub key (qux vs. foo) => partial new queries
    assert len(responses.calls) == 1


def test_json_cache():
    cache_dir = tempfile.mkdtemp()

    client = Client(f"{EXAMPLE_URL}", "token")
    loader = Loader(client, cache_dir=cache_dir)

    calls = 0

    def foo():
        nonlocal calls
        calls += 1
        return {"foo": "bar"}

    assert loader.with_cache("json", ["foo"], foo) == {"foo": "bar"}
    assert loader.with_cache("json", ["foo"], foo) == {"foo": "bar"}
    assert calls == 1

    calls = 0
    # missing data in keys => new call
    assert loader.with_cache("json", ["foo", None], foo) == {"foo": "bar"}
    assert calls == 1


def mock_query(data):
    responses.get(
        f"{EXAMPLE_URL}/api/v1/query",
        json={"status": "success", "data": {"result": data}},
        status=200,
    )


def mock_instant_query():
    builder = PromInstantDatasetBuilder(TEST_TIME)
    builder.ts({"foo": "bar"}).value(42)
    builder.ts({"foo": "baz"}).value(lambda t: t.minute)

    mock_query(builder.build_raw())


def mock_range_query():
    builder = PromRangeDatasetBuilder(TEST_TIME - timedelta(hours=1), TEST_TIME)
    builder.ts({"foo": "bar"}).interval(timedelta(minutes=2), timedelta(minutes=4), 42)
    builder.ts({"foo": "baz"}).interval(
        timedelta(minutes=3), timedelta(minutes=5), lambda t: t.minute
    )

    mock_query(builder.build_raw())


def requests_queries():
    ret = []

    for call in responses.calls:
        url = urlparse(call.request.url)
        qs = parse_qs(url.query)
        ret.append(qs)

    return ret
