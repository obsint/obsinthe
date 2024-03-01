from datetime import datetime
import requests_mock

from obsinthe.prometheus.client import Client


EXAMPLE_URL = "http://prometheus.example.com"


def test_query():
    with requests_mock.mock() as m:
        m.get(
            f"{EXAMPLE_URL}/api/v1/query",
            json={"status": "success", "data": {"result": [1, 2, 3]}},
        )
        client = Client(f"{EXAMPLE_URL}", "token")
        assert client.query("up") == [1, 2, 3]


def test_query_range():
    with requests_mock.mock() as m:
        m.get(
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
        assert m.last_request.qs == {
            "end": ["1704114000"],
            "query": ["up"],
            "start": ["1704106800"],
            "step": ["60"],
        }


def test_check_connection():
    with requests_mock.mock() as m:
        m.get(
            f"{EXAMPLE_URL}/",
            json={"status": "success", "data": {"result": [1, 2, 3]}},
        )
        client = Client(f"{EXAMPLE_URL}", "token")
        assert client.check_connection() == True

        m.get(
            f"{EXAMPLE_URL}/",
            json={"status": "Forbidden"},
            status_code=403,
        )
        assert client.check_connection() == False


def test_all_metrics():
    with requests_mock.mock() as m:
        m.get(
            f"{EXAMPLE_URL}/api/v1/label/__name__/values",
            json={"status": "success", "data": [1, 2, 3]},
        )
        client = Client(f"{EXAMPLE_URL}", "token")
        assert client.all_metrics() == [1, 2, 3]
