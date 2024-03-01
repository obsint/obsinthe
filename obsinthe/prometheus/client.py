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
    A Class for collection of metrics from a Prometheus Host.

    :param url: (str) url for the prometheus host
    :param headers: (dict) A dictionary of http headers to be used to communicate with
        the host. Example: {"Authorization": "bearer my_oauth_token_to_the_host"}
    :param disable_ssl: (bool) If set to True, will disable ssl certificate verification
        for the http requests made to the prometheus host
    :param retry: (Retry) Retry adapter to retry on HTTP errors
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
        """Send a GET request to the Prometheus Host."""
        return self._session.get(
            f"{self.url}{path}",
            verify=self.ssl_verification,
            headers=self.headers,
            params=params,
        )

    def check_prometheus_connection(self, params: Optional[dict] = None) -> bool:
        """
        Check Promethus connection.

        :param params: (dict) Optional dictionary containing parameters to be
            sent along with the API request.
        :returns: (bool) True if the endpoint can be reached, False if cannot be reached.
        """
        response = self.get("/", params or {})
        return response.ok

    def all_metrics(self, params: Optional[dict] = None):
        """
        Get the list of all the metrics that the prometheus host scrapes.

        :param params: (dict) Optional dictionary containing GET parameters to be
            sent along with the API request, such as "time"
        :returns: (list) A list of names of all the metrics available from the
            specified prometheus host
        :raises:
            (RequestException) Raises an exception in case of a connection error
            (PrometheusApiClientException) Raises in case of non 200 response status code
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
        Send a custom query to a Prometheus Host.

        This method takes as input a string which will be sent as a query to
        the specified Prometheus Host. This query is a PromQL query.

        :param query: (str) This is a PromQL query, a few examples can be found
            at https://prometheus.io/docs/prometheus/latest/querying/examples/
        :param params: (dict) Optional dictionary containing GET parameters to be
            sent along with the API request, such as "time"
        :returns: (list) A list of metric data received in response of the query sent
        :raises:
            (RequestException) Raises an exception in case of a connection error
            (PrometheusApiClientException) Raises in case of non 200 response status code
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
                "HTTP Status Code {} ({!r})".format(
                    response.status_code, response.content
                )
            )

    def query_range(
        self,
        query: str,
        start_time: datetime,
        end_time: datetime,
        step: str,
        params: dict = None,
    ):
        """
        Send a query_range to a Prometheus Host.

        This method takes as input a string which will be sent as a query to
        the specified Prometheus Host. This query is a PromQL query.

        :param query: (str) This is a PromQL query, a few examples can be found
            at https://prometheus.io/docs/prometheus/latest/querying/examples/
        :param start_time: (datetime) A datetime object that specifies the query range start time.
        :param end_time: (datetime) A datetime object that specifies the query range end time.
        :param step: (str) Query resolution step width in duration format or float number of seconds
        :param params: (dict) Optional dictionary containing GET parameters to be
            sent along with the API request, such as "timeout"
        :returns: (dict) A dict of metric data received in response of the query sent
        :raises:
            (RequestException) Raises an exception in case of a connection error
            (PrometheusApiClientException) Raises in case of non 200 response status code
        """
        if params:
            params = params.copy()
        else:
            params = {}

        params["query"] = query
        params["start"] = round(start_time.timestamp())
        params["end"] = round(end_time.timestamp())
        params["step"] = step

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
