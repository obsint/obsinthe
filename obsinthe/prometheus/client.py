# Initially taken from https://github.com/4n4nd/prometheus-api-client-python
# /blob/master/prometheus_api_client/prometheus_connect.py
# but reworked to fit better our use cases.

import logging
from datetime import datetime
from typing import Optional
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry


class PrometheusApiClientException(Exception):
    pass


# set up logging

_LOGGER = logging.getLogger(__name__)

# In case of a connection failure try 2 more times
MAX_REQUEST_RETRIES = 3
# wait 1 second before retrying in case of an error
RETRY_BACKOFF_FACTOR = 1
# retry only on these status
RETRY_ON_STATUS = [408, 429, 500, 502, 503, 504]


class Client:
    """
    A class for collecting metrics from a Prometheus host.

    Args:
        url (str): The URL for the Prometheus host.
        headers (dict): A dictionary of HTTP headers.
        disable_ssl (bool): Disable SSL verification. DON'T USE IN PRODUCTION,
            and better DON'T USE AT ALL.
        retry (Retry): Retry adapter to handle retries on HTTP errors.
    """

    def __init__(
        self,
        url: str,
        token: str,
        disable_ssl: bool = False,
        retry: Retry = None,
    ):
        """Functions as a Constructor for the class PrometheusConnect."""
        if url is None:
            raise TypeError("missing url")

        self.url = url
        self.token = token

        self.headers = {"Authorization": f"bearer {self.token}"}
        self.prometheus_host = urlparse(self.url).netloc
        self._all_metrics = None
        self.ssl_verification = not disable_ssl

        if retry is None:
            retry = Retry(
                total=MAX_REQUEST_RETRIES,
                backoff_factor=RETRY_BACKOFF_FACTOR,
                status_forcelist=RETRY_ON_STATUS,
            )

        self._session = requests.Session()
        self._session.mount(self.url, HTTPAdapter(max_retries=retry))

    def get(self, path, params=None):
        """Send a generic GET request."""
        return self._session.get(
            f"{self.url}{path}",
            verify=self.ssl_verification,
            headers=self.headers,
            params=params,
        )

    def check_prometheus_connection(self, params: Optional[dict] = None) -> bool:
        """
        Check Promethus connection.

        Args:
            params: (dict) Additional request arguments.

        Returns:
            bool: True if the endpoint can be reached, False otherwise.
        """
        response = self.get("/", params or {})
        return response.ok

    def all_metrics(self, params: Optional[dict] = None):
        """
        Get the list of all available metrics.

        Args:
            params: (dict) Additional request arguments.

        Returns:
            list: A list of metrics names.

        Raises:
            RequestException: If a connection error occurs.
            PrometheusApiClientException: If a non-200 response is received.
        """
        response = self.get("/api/v1/label/__name__/values", params or {})

        if response.status_code == 200:
            self._all_metrics = response.json()["data"]
        else:
            raise PrometheusApiClientException(
                "HTTP Status Code {} ({!r})".format(
                    response.status_code, response.content
                )
            )
        return self._all_metrics

    def query(
        self, query: str, time: Optional[datetime] = None, params: Optional[dict] = None
    ):
        """
        Send a custom PromQL query.

        Args:
            query (str): The PromQL query string.
            time (datetime, optional): The time to query.
            params (dict, optional): Additional request arguments.

        Returns:
            list: A list of metric data received in response of the query sent.

        Raises:
            RequestException: If a connection error occurs.
            PrometheusApiClientException: If a non-200 response is received.
        """
        if params:
            params = params.copy()
        else:
            params = {}

        params["query"] = query
        if time:
            params["time"] = int(time.timestamp())

        response = self.get("/api/v1/query", params)

        if response.status_code == 200:
            return response.json()["data"]["result"]
        else:
            raise PrometheusApiClientException(
                "HTTP Status Code {} ({!r})".format(respon)
            )

    def query_range(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        step: int,
        params: dict = None,
    ):
        """
        Sends a PromQL range query.

        Args:
            query (str): PromQL query string.
            start_time (datetime): Range query start time.
            end_time (datetime): Range query end time.
            step (int): Number of seconds.
            params (dict, optional): Additional query params for the request.

        Returns:
            dict: A dictionary containing metric data received in response.

        Raises:
            RequestException: If a connection error occurs.
            PrometheusApiClientException: If a non-200 response status code is received.
        """
        if params:
            params = params.copy()
        else:
            params = {}

        params["query"] = query
        params["start"] = round(start_time.timestamp())
        params["end"] = round(end_time.timestamp())
        params["step"] = str(step)

        response = self.get(
            "/api/v1/query_range",
            params=params,
        )
        if response.status_code == 200:
            return response.json()["data"]["result"]
        else:
            raise PrometheusApiClientException(
                "HTTP Status Code {} ({!r})".format(
                    response.status_code, response.content
                )
            )
