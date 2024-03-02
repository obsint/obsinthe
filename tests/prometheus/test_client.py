from datetime import datetime
from urllib.parse import urlparse, parse_qs

import pytest
import responses

from obsinthe.prometheus.client import Client, PrometheusApiClientException


EXAMPLE_URL = "http://prometheus.example.com"


@responses.activate()
def test_query():
    responses.get(
        f"{EXAMPLE_URL}/api/v1/query",
        json={"status": "success", "data": {"result": [1, 2, 3]}},
        status=200,
    )

    client = Client(f"{EXAMPLE_URL}", "token")

    assert client.query("up") == [1, 2, 3]


@responses.activate()
def test_query_retry():
    url = f"{EXAMPLE_URL}/api/v1/query"
    rsp1 = responses.get(url, json={"status": "error"}, status=500)
    rsp2 = responses.get(
        url, json={"status": "success", "data": {"result": [1, 2, 3]}}, status=200
    )

    client = Client(f"{EXAMPLE_URL}", "token")

    assert client.query("up") == [1, 2, 3]
    assert rsp1.call_count == 1
    assert rsp2.call_count == 1

    for _ in range(4):
        responses.get(url, json={"status": "error"}, status=500)

    pytest.raises(PrometheusApiClientException, client.query, "up")


@responses.activate()
def test_query_range():
    responses.get(
        f"{EXAMPLE_URL}/api/v1/query_range",
        json={"status": "success", "data": {"result": [1, 2, 3]}},
    )

    client = Client(EXAMPLE_URL, "token")

    assert client.query_range(
        "up",
        start_time=datetime(2024, 1, 1, 12),
        end_time=datetime(2024, 1, 1, 14),
        step=60,
    ) == [1, 2, 3]

    url = urlparse(responses.calls[0].request.url)
    qs = parse_qs(url.query)

    assert qs == {
        "end": ["1704114000"],
        "query": ["up"],
        "start": ["1704106800"],
        "step": ["60"],
    }


@responses.activate()
def test_check_connection():
    responses.get(
        f"{EXAMPLE_URL}/",
        json={"status": "success", "data": {"result": [1, 2, 3]}},
    )

    client = Client(f"{EXAMPLE_URL}", "token")

    assert client.check_connection() == True

    responses.get(
        f"{EXAMPLE_URL}/",
        json={"status": "Forbidden"},
        status=403,
    )

    assert client.check_connection() == False


@responses.activate()
def test_all_metrics():
    responses.get(
        f"{EXAMPLE_URL}/api/v1/label/__name__/values",
        json={"status": "success", "data": [1, 2, 3]},
    )

    client = Client(f"{EXAMPLE_URL}", "token")

    assert client.all_metrics() == [1, 2, 3]
