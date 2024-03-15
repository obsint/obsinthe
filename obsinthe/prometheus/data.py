from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from tqdm.auto import tqdm


# Default resolution for Prometheus queries, in seconds.
DEFAULT_RESOLUTION = timedelta(minutes=5)

DatasetType = Union[Type["InstantDataset"], Type["RangeDataset"]]


class DatasetBase:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __repr__(self):
        return (
            f"{self.__class__.__name__} with columns {list(self.df.columns)} "
            f"and {len(self.df)} rows"
        )

    def fmap(self, fn):
        return type(self)(fn(self.df))

    def query(self, *args, **kwargs):
        return self.fmap(lambda df: df.query(*args, **kwargs))

    @staticmethod
    def from_raw(raw_data, columns=None) -> Optional["DatasetBase"]:
        raise NotImplementedError


class InstantDataset(DatasetBase):
    @staticmethod
    def from_raw(raw_data, columns=None) -> Optional["InstantDataset"]:
        if len(raw_data) == 0:
            return None
        if "value" not in raw_data[0]:
            return None

        data = [
            {
                **extract_columns_data(d["metric"], columns),
                "timestamp": d["value"][0],
                "value": float(d["value"][1]),
            }
            for d in raw_data
        ]
        df = pd.DataFrame(data)
        if df.empty:
            return None
        df["timestamp"] = pd.to_datetime(df["timestamp"] * 1e9)
        return InstantDataset(df)


class RangeDataset(DatasetBase):
    @staticmethod
    def from_raw(raw_data, columns=None) -> Optional["RangeDataset"]:
        if len(raw_data) == 0:
            return None
        if "values" not in raw_data[0]:
            return None

        def typecast_values(values):
            # we use 1-d array by default to avoid issues with parquet conversion
            return np.array([[int(t), float(v)] for (t, v) in values]).flatten()

        data = [
            {
                **extract_columns_data(i["metric"], columns),
                "values": typecast_values(i["values"]),
            }
            for i in raw_data
        ]

        df = pd.DataFrame(data=data)
        if df.empty:
            return None

        return RangeDataset(df)

    def to_range_intervals_ds(
        self, threshold: timedelta = DEFAULT_RESOLUTION
    ) -> "RangeIntervalsDataset":
        """Convert into RangeIntervalsDataset.

        It doesn't use the TS values, just determines the intervals for times
        the values were present.

        It preserves the original rows, just replaces the values column with the
        intervals.

        Args:
            threshold (int, optional): The threshold value used to determine the
            intervals in seconds. Defaults to `DEFAULT_RESOLUTION`.

        Returns:
            RangeIntervalsDataset: A new RangeIntervalsDataset instance.
        """

        df = self.df.copy()

        threshold_s = threshold.total_seconds()

        # we use vs[0::2] as we care only about timestamps: assuming values being all
        # ones as in alerts case.
        intervals = df["values"].apply(
            lambda vs: np_timestamps_to_intervals(vs[0::2], threshold_s)
        )

        df["intervals"] = intervals
        df.drop(columns=["values"], inplace=True)
        return RangeIntervalsDataset(df)

    def to_intervals_ds(self, threshold: timedelta) -> "IntervalsDataset":
        """Convert into IntervalsDataset.

        It doesn't use the TS values, just determines the intervals for times
        the values were present.

        It preserves the original rows, just replaces the values column with the
        intervals.

        Returns:
            IntervalsDataset: A new IntervalsDataset instance.
        """
        return self.to_range_intervals_ds(threshold).to_intervals_ds()


class RangeIntervalsDataset(DatasetBase):
    def to_intervals_ds(self) -> "IntervalsDataset":
        """Expand into a IntervalsDataset.

        Each interval is represented as a row in the DataFrame, with the start
        and end times as columns.

        As a result, the number of rows in the DataFrame will increase.

        Returns:
            Corresponding IntervalsDataset instance.
        """
        # explode the intervals into separate rows
        df = self.df.explode("intervals", ignore_index=True)

        # split the intervals into start and end columns
        intervals = pd.DataFrame(
            data={
                "start": df["intervals"].apply(
                    lambda v: datetime.utcfromtimestamp(v[0]).replace(
                        tzinfo=timezone.utc
                    )
                ),
                "end": df["intervals"].apply(
                    lambda v: datetime.utcfromtimestamp(v[1]).replace(
                        tzinfo=timezone.utc
                    )
                ),
            }
        )

        # drop the original values and intervals columns
        df.drop(columns=["intervals"], inplace=True)

        df = pd.concat([df, intervals], axis=1)
        return IntervalsDataset(df)


class IntervalsDataset(DatasetBase):
    def merge_overlaps(self, threshold=timedelta(0), columns=None):
        """Merge overlapping time intervals.

        Considering a list of time-intervals at the input, merge the rows
        that have the same columns values + have some time overlaps.
        """
        if self.df.empty:
            return self

        df = self.df

        def _identify_intervals(sub_df):
            if len(sub_df) == 1:
                sub_df["interval_label"] = 0
                return sub_df
            previous_ends = [None]
            for i in range(1, len(sub_df)):
                previous_end = sub_df.iloc[i - 1]["end"]
                if i > 1:
                    # handle one interval overlapping multiple others, taking
                    # always the highest from the previous ends
                    previous_end = max(previous_end, previous_ends[i - 1])
                previous_ends.append(previous_end)

            sub_df["previous_end"] = previous_ends
            period_start = (sub_df["start"] - sub_df["previous_end"]) > threshold
            sub_df["interval_label"] = period_start.cumsum()
            return sub_df

        if columns is None:
            columns = list(df.columns.drop(["start", "end"]))
        df = df.sort_values("start")

        interval_labels = df.groupby(
            columns, observed=True, as_index=False, dropna=False, group_keys=False
        ).apply(_identify_intervals)

        intervals = interval_labels.groupby(
            columns + ["interval_label"],
            observed=True,
            as_index=True,
            dropna=False,
            group_keys=True,
        ).agg(start=("start", "min"), end=("end", "max"))
        intervals["sub_intervals"] = interval_labels.groupby(
            columns + ["interval_label"],
            observed=True,
            as_index=True,
            dropna=False,
            group_keys=True,
        ).apply(lambda df: list(zip(df["start"], df["end"])))

        intervals.reset_index(inplace=True)
        intervals.drop("interval_label", axis=1, inplace=True)

        return IntervalsDataset(intervals)


class DatasetCollection:
    def __init__(self, datasets: List[DatasetBase]):
        self.datasets = datasets

    def __getitem__(self, idx):
        return self.datasets[idx]

    def __len__(self):
        return len(self.datasets)

    def __iter__(self):
        return iter(self.datasets)

    def __repr__(self):
        return f"DatasetCollection with {len(self.datasets)} datasets"

    def fmap(self, fn):
        return DatasetCollection([fn(ds) for ds in self.datasets])

    def query(self, *args, **kwargs):
        return self.fmap(lambda ds: ds.query(*args, **kwargs))


def extract_columns_data(raw_data, columns):
    if columns is None:
        return raw_data

    ret, extra = {}, {}

    for k, v in raw_data.items():
        if k in columns:
            ret[k] = v
        else:
            extra[k] = v
    ret["extra"] = extra

    return ret


def raw_to_ds(
    raw_data,
    ds_type: Optional[DatasetBase] = None,
    columns: Optional[list] = None,
) -> Optional[DatasetBase]:
    if ds_type is not None:
        return ds_type.from_raw(raw_data)

    for ts_type in [RangeDataset, InstantDataset]:
        ds = ts_type.from_raw(raw_data, columns)
        if ds is not None:
            return ds


def np_timestamps_to_intervals(
    timestamps: np.ndarray, threshold: float
) -> List[Tuple[datetime, datetime]]:
    """Convert a list of timestamps into a set of intervals.

    Args:
        timestamps (np.ndarray): A NumPy array containing the timestamps.
        threshold (float): The threshold value used to determine the intervals.

    Returns:
        List[Tuple[int, int]]: A set of intervals represented as tuples of
        (start_time, end_time).

    Example:
        >>> timestamps = np.array([0, 1, 2, 5, 10, 12, 15, 20])
        >>> threshold = 2
        >>> convert_to_intervals(timestamps, threshold)
        [(0, 2), (5, 5), (10, 12), (15, 15), (20, 20)]

    Notes:
        - Timestamps should be sorted in ascending order before passing to this
          function.
        - Intervals are inclusive of the start time and exclusive of the end time.
        - Consecutive timestamps with a difference less than or equal to the
          threshold will be grouped into the same interval.
        - If the difference between two consecutive timestamps is greater than
          the threshold,
          a new interval will be created.
        - The intervals use utc datetime.
    """
    differences = np.diff(timestamps)
    split_indices = np.where(differences > threshold)[0] + 1
    groups = np.split(timestamps, split_indices)
    return [(g[0], g[-1]) for g in groups]


def identify_intervals(df, resolution, time_column="timestamp"):
    df = df.sort_values(time_column)
    ts = df[time_column]
    if is_datetime(ts):
        ts = ts.astype(int) // 1e9
    period_start = ts.diff() > resolution.total_seconds()
    df["interval_label"] = period_start.cumsum()
    return df


def intervals_concat_days(ds_collection, threshold=timedelta(0)):
    ret_dfs = []
    to_merge_left = None
    for day_ds in tqdm(ds_collection.datasets):
        day_df = day_ds.df
        # Merge the left over from the previous day.

        if to_merge_left is not None:
            # We consider only beginning of the day.
            to_merge_right = day_df.loc[
                # In daily, we consider only beginning of the day.
                ((day_df["start"] - day_df["start"].dt.normalize()) <= threshold)
            ]

            # merge the intervals
            merged_df = (
                IntervalsDataset(
                    pd.concat([to_merge_left, to_merge_right], ignore_index=True)
                )
                .merge_overlaps(
                    threshold=threshold,
                )
                .df
            )
            if "sub_intervals" in merged_df.columns:
                merged_df.drop("sub_intervals", axis=1, inplace=True)

            # Replace the intervals in to_merge_right with the merged ones.
            day_df = day_df.loc[day_df.index.difference(to_merge_right.index)]
            day_df = pd.concat([day_df, merged_df], ignore_index=True)

        # Prepare the left over for the next day.
        to_merge_left = day_df.loc[
            # End of day.
            (
                (day_df["end"] - day_df["end"].dt.normalize())
                >= (timedelta(days=1) - threshold)
            )
            # Special case of end right at the midnight.
            | ((day_df["end"] - day_df["end"].dt.normalize()) == timedelta(0))
        ]

        # Add the rest of the day_df to the ret_dfs: no need to merge with anything.
        day_df = day_df.loc[day_df.index.difference(to_merge_left.index)]
        ret_dfs.append(day_df)

    # Finalizing the left over from the last day.
    ret_dfs.append(to_merge_left)
    df = pd.concat(ret_dfs, ignore_index=True)
    return IntervalsDataset(df)


def group_by_time(
    df,
    time_column,
    extra_groupby_columns=None,
    tolerance=timedelta(0),
    group_id_column="group_id",
) -> pd.DataFrame:
    if extra_groupby_columns:
        groups = df.groupby(extra_groupby_columns, as_index=False).apply(
            lambda df: identify_intervals(df, tolerance, time_column)
        )
    else:
        groups = identify_intervals(df, tolerance, time_column)

    if extra_groupby_columns:
        groups[group_id_column] = groups.apply(
            lambda r: "-".join(
                [str(v) for v in r[[*extra_groupby_columns, "interval_label"]].values]
            ),
            axis=1,
        )
        groups.drop("interval_label", axis=1, inplace=True)
    else:
        groups.rename(columns={"interval_label": group_id_column}, inplace=True)

    return groups.reset_index(drop=True)


def one_hot_encode(df, index, column):
    return df.assign(value=1).pivot_table(
        index=index, columns=column, values="value", fill_value=0
    )
