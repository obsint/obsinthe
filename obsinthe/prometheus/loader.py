import json
import os
from hashlib import sha256

import pandas as pd
from tqdm.auto import tqdm

from obsinthe.utils import time as time_utils

from typing import Callable, Dict, Iterable, Optional, Union
from .client import Client
from .data import raw_to_df
from .data import TimeSeriesType


def digest(string):
    return sha256(string.encode("utf8")).hexdigest()


class JsonFileCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def with_cache(self, cache_key: list, func: Callable):
        cache_file = os.path.join(self.cache_dir, *cache_key, "data.json")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if os.path.exists(cache_file):
            with open(cache_file) as f:
                return json.load(f)
        else:
            data = func()
            with open(cache_file, "w") as f:
                json.dump(data, f)
            return data


class ParquetFileCache:
    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def with_cache(self, cache_key: list, func: Callable):
        cache_file = os.path.join(self.cache_dir, *cache_key, "data.parquet")
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        if os.path.exists(cache_file):
            return pd.read_parquet(cache_file, engine="pyarrow")
        else:
            data = func()
            if not isinstance(data, pd.DataFrame):
                raise ValueError("Data must be a DataFrame")
            data.to_parquet(cache_file, engine="pyarrow")
            return data


class Loader:
    def __init__(self, client: Client, cache_dir: Optional[str] = None):
        self.client = client
        if cache_dir:
            self.json_cache = JsonFileCache(cache_dir)
            self.parquet_cache = ParquetFileCache(cache_dir)
        else:
            self.json_cache = None
            self.parquet_cache = None

    def query(
        self, query, time, ts_type: Optional[TimeSeriesType] = None, cache_key=None
    ):
        if cache_key:
            if not isinstance(cache_key, list):
                cache_key = [cache_key, time.isoformat()]
        return self.with_cache(
            "parquet",
            cache_key,
            lambda: raw_to_df(self.client.query(query, time=time), ts_type=ts_type),
        )

    def interval_query(
        self,
        query: str,
        start: str,
        end: str,
        cache_key: Optional[str] = None,
        ts_type: Optional[TimeSeriesType] = None,
    ) -> list[pd.DataFrame]:
        # load raw data
        data = []

        for _, interval_end in tqdm(time_utils.gen_daily_intervals(start, end)):
            d = self.query(query, interval_end, ts_type, cache_key)
            data.append(d)

        return data

    def batch_query(
        self,
        inputs: Dict[pd.Timestamp, Iterable],
        query_template: Union[str, Callable],
        cache_key: Optional[str] = None,
        batch_size: int = 500,
        ts_type: Optional[TimeSeriesType] = None,
    ):
        def format_query(items):
            if isinstance(query_template, str):
                return query_template % "|".join(items)
            else:
                return query_template(items)

        keys = sorted(list(inputs.keys()))

        data = []
        for interval_end in tqdm(keys):
            items = inputs[interval_end]
            items = sorted(items)
            day_data = []

            for pos in range(0, len(items), batch_size):
                sub_items = items[pos : (pos + batch_size)]  # noqa: E203
                query = format_query(sub_items)

                d = self.query(
                    query,
                    interval_end,
                    ts_type,
                    [cache_key, interval_end.isoformat(), sub_items],
                )
                if d is not None:
                    day_data.append(d)

            if day_data:
                df = pd.concat(day_data, ignore_index=True).copy()
                if "timestamp" not in df.columns:  # in case of range df
                    df["timestamp"] = pd.to_datetime(interval_end)
                data.append(df)

        return data

    def with_cache(self, format: str, key: Optional[list], func: Callable):
        if format == "json":
            cache = self.json_cache
        elif format == "parquet":
            cache = self.parquet_cache
        else:
            raise ValueError("Invalid cache format")

        if cache and key and all(key):
            # convert all list elements to digested string
            key = [digest("".join(k)) if isinstance(k, list) else k for k in key]

            return cache.with_cache(key, func)
        else:
            return func()
