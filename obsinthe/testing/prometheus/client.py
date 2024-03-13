from datetime import datetime
from typing import Optional
from typing import Tuple

from responses import RequestsMock

from obsinthe.prometheus.client import Client
from obsinthe.testing.prometheus.builder import normalize_tz
from obsinthe.testing.prometheus.builder import PromDatasetBuilderBase
from obsinthe.utils.time import gen_daily_intervals


class MockedClient(Client):
    def __init__(
        self,
        builder: PromDatasetBuilderBase,
    ):
        self.url = "https://prometheus.example.com"
        super().__init__(self.url, "token")
        self.builder = builder

    def mock_setup(self, interval: Optional[Tuple[datetime, datetime]] = None):
        self.mock = RequestsMock(assert_all_requests_are_fired=False)
        self.set_from_builder(self.mock, self.builder, interval)

    def set_from_builder(
        self,
        mock,
        builder: PromDatasetBuilderBase,
        interval: Optional[Tuple[datetime, datetime]] = None,
    ):
        if interval:
            start, end = interval
            start, end = normalize_tz(start), normalize_tz(end)
            for s, e in gen_daily_intervals(start, end):
                mock.get(
                    f"{self.url}/api/v1/query",
                    json={
                        "status": "success",
                        "data": {"result": builder.build_raw(s, e)},
                    },
                )
            pass
        else:
            mock.get(
                f"{self.url}/api/v1/query",
                json={"status": "success", "data": {"result": builder.build_raw()}},
            )

    def get(self, path, params=None):
        self.mock.start()
        try:
            return super().get(path, params)
        finally:
            self.mock.stop()
