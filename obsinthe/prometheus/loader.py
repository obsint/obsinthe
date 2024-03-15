import json
import os
from hashlib import sha256
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Optional
from typing import Union

import pandas as pd
from tqdm.auto import tqdm

from obsinthe.utils import time as time_utils

from .client import Client
from .data import DatasetCollection
from .data import DatasetType
from .data import InstantDataset
from .data import RangeDataset
from .data import raw_to_ds


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
    DS_TYPES = {
        "InstantDataset": InstantDataset,
        "RangeDataset": RangeDataset,
        "DataFrame": pd.DataFrame,
    }

    def __init__(self, cache_dir: str):
        self.cache_dir = cache_dir

    def with_cache(self, cache_key: list, func: Callable):
        pq_cache_file = os.path.join(self.cache_dir, *cache_key, "data.parquet")
        type_cache_file = os.path.join(self.cache_dir, *cache_key, "data.type")
        os.makedirs(os.path.dirname(pq_cache_file), exist_ok=True)
        ret = self.try_read(pq_cache_file, type_cache_file)
        if ret is not None:
            return ret
        else:
            data = func()
            self.write(pq_cache_file, type_cache_file, data)
            return data

    def try_read(self, pq_cache_file, type_cache_file):
        if not os.path.exists(pq_cache_file) or not os.path.exists(type_cache_file):
            return None

        df = pd.read_parquet(pq_cache_file, engine="pyarrow")
        with open(type_cache_file) as f:
            ds_type_str = f.read()
            ds_type = self.DS_TYPES[ds_type_str]

        if ds_type == pd.DataFrame:
            return df
        else:
            return ds_type(df)

    def write(self, pq_cache_file, type_cache_file, data):
        if data.__class__.__name__ not in self.DS_TYPES:
            raise ValueError("Invalid data type")

        if not isinstance(data, pd.DataFrame):
            df = data.df
        else:
            df = data
        df.to_parquet(pq_cache_file, engine="pyarrow")
        with open(type_cache_file, "w") as f:
            f.write(data.__class__.__name__)


class Loader:
    def __init__(self, client: Client, cache_dir: Optional[str] = None):
        self.client = client
        if cache_dir:
            self.json_cache = JsonFileCache(cache_dir)
            self.parquet_cache = ParquetFileCache(cache_dir)
        else:
            self.json_cache = None
            self.parquet_cache = None

    def query(self, query, time, ds_type: Optional[DatasetType] = None, cache_key=None):
        if cache_key:
            if not isinstance(cache_key, list):
                cache_key = [cache_key, time.isoformat()]
        return self.with_cache(
            "parquet",
            cache_key,
            lambda: raw_to_ds(self.client.query(query, time=time), ds_type=ds_type),
        )

    def interval_query(
        self,
        query: str,
        start: str,
        end: str,
        cache_key: Optional[str] = None,
        ds_type: Optional[DatasetType] = None,
    ) -> DatasetCollection:
        # load raw data
        datasets = []

        for _, interval_end in tqdm(time_utils.gen_daily_intervals(start, end)):
            ds = self.query(query, interval_end, ds_type, cache_key)
            datasets.append(ds)

        return DatasetCollection(datasets)

    def batch_query(
        self,
        inputs: Dict[pd.Timestamp, Iterable],
        query_template: Union[str, Callable],
        cache_key: Optional[str] = None,
        batch_size: int = 500,
        ds_type: Optional[DatasetType] = None,
        post_process: Optional[Callable] = None,
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

                cache_keys = [cache_key, interval_end.isoformat(), sub_items]

                ds = self.query(
                    query,
                    interval_end,
                    ds_type,
                    cache_keys,
                )
                if ds is not None:
                    # make sure the ds_type is determined: will be used later.
                    ds_type = type(ds)
                if post_process:
                    ds = self.with_cache(
                        "parquet", cache_keys + ["post"], lambda: post_process(ds)
                    )

                if ds is not None:
                    day_data.append(ds)

            if day_data:
                day_data_dfs = [ds.df for ds in day_data]
                df = pd.concat(day_data_dfs, ignore_index=True).copy()
                if "timestamp" not in df.columns:  # in case of range df
                    df["timestamp"] = pd.to_datetime(interval_end)
                data.append(ds_type(df))

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
