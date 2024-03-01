from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple, Type, Union
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from pandas.api.types import is_datetime64_any_dtype as is_datetime

# Default resolution for Prometheus queries, in seconds.
DEFAULT_RESOLUTION = 60 * 5

TimeSeriesType = Union[Type["InstantDF"], Type["RangeDF"]]


# Placeholder classes for specifying formats of the dataframes.
# In the future, objects could actually contain the logic. For now, other
# functions will be responsible for converting the dataframes to these formats
# and return plain dataframes.
class TimeSeriesDF:
    @staticmethod
    def detect_from_raw(raw_data) -> TimeSeriesType:
        if len(raw_data) > 0 and "values" in raw_data[0]:
            return RangeDF
        else:
            return InstantDF


class InstantDF(TimeSeriesDF):
    pass


class RangeDF(TimeSeriesDF):
    pass


def raw_to_instant_df(raw_data):
    data = [
        {**d["metric"], "timestamp": d["value"][0], "value": float(d["value"][1])}
        for d in raw_data
    ]
    df = pd.DataFrame(data)
    if df.empty:
        return None
    df["timestamp"] = pd.to_datetime(df["timestamp"] * 1e9)
    return df


def raw_to_range_df(raw_data):
    def typecast_values(values):
        # we use 1-d array by default to avoid issues with parquet conversion
        return np.array([[int(t), float(v)] for (t, v) in values]).flatten()

    data = [{**i["metric"], "values": typecast_values(i["values"])} for i in raw_data]

    return pd.DataFrame(data=data)


def raw_to_df(
    raw_data, ts_type: Optional[TimeSeriesType] = None
) -> Optional[pd.DataFrame]:
    if ts_type is None:
        ts_type = TimeSeriesDF.detect_from_raw(raw_data)

    if issubclass(ts_type, RangeDF):
        return raw_to_range_df(raw_data)
    else:
        return raw_to_instant_df(raw_data)


def np_timestamps_to_intervals(
    timestamps: np.ndarray, threshold: float
) -> List[Tuple[datetime, datetime]]:
    """Convert a list of timestamps into a set of intervals.

    Args:
        timestamps (np.ndarray): A NumPy array containing the timestamps.
        threshold (float): The threshold value used to determine the intervals.

    Returns:
        List[Tuple[int, int]]: A set of intervals represented as tuples of (start_time, end_time).

    Example:
        >>> timestamps = np.array([0, 1, 2, 5, 10, 12, 15, 20])
        >>> threshold = 2
        >>> convert_to_intervals(timestamps, threshold)
        [(0, 2), (5, 5), (10, 12), (15, 15), (20, 20)]

    Notes:
        - Timestamps should be sorted in ascending order before passing to this function.
        - Intervals are inclusive of the start time and exclusive of the end time.
        - Consecutive timestamps with a difference less than or equal to the threshold
          will be grouped into the same interval.
        - If the difference between two consecutive timestamps is greater than the threshold,
          a new interval will be created.
        - The intervals use utc datetime.
    """
    differences = np.diff(timestamps)
    split_indices = np.where(differences > threshold)[0] + 1
    groups = np.split(timestamps, split_indices)
    return [(g[0], g[-1]) for g in groups]


def range_df_to_range_intervals_df(
    df: pd.DataFrame, threshold: int = DEFAULT_RESOLUTION
) -> pd.DataFrame:
    """Convert a DataFrame containing ranges into a DataFrame containing intervals.

    It preserves the original rows, just replaces the values column with the intervals.

    Args:
        df (pd.DataFrame): A DataFrame containing ranges.
        threshold (int, optional): The threshold value used to determine the intervals.
            Defaults to DEFAULT_RESOLUTION.

    Returns:
        pd.DataFrame: A DataFrame containing intervals.
    """

    df = df.copy()

    # we use vs[0::2] as we care only about timestamps.
    intervals = df["values"].apply(
        lambda vs: np_timestamps_to_intervals(vs[0::2], threshold)
    )

    df["intervals"] = intervals
    df.drop(columns=["values"], inplace=True)
    return df


def range_intervals_df_to_intervals_df(df: pd.DataFrame) -> pd.DataFrame:
    """Expand a DataFrame containing compact intervals into a DataFrame containing intervals.

    Each interval is represented as a row in the DataFrame, with the start and end times as columns.

    As a result, the number of rows in the DataFrame will increase.

    Args:
        df (pd.DataFrame): A DataFrame containing compact intervals.

    Returns:
        pd.DataFrame: A DataFrame containing intervals.
    """
    # explode the intervals into separate rows
    df = df.explode("intervals", ignore_index=True)

    # split the intervals into start and end columns
    intervals = pd.DataFrame(
        data={
            "start": df["intervals"].apply(
                lambda v: datetime.utcfromtimestamp(v[0]).replace(tzinfo=timezone.utc)
            ),
            "end": df["intervals"].apply(
                lambda v: datetime.utcfromtimestamp(v[1]).replace(tzinfo=timezone.utc)
            ),
        }
    )

    # drop the original values and intervals columns
    df.drop(columns=["intervals"], inplace=True)

    return pd.concat([df, intervals], axis=1)


def range_df_to_intervals_df(
    range_df: pd.DataFrame, threshold: int = DEFAULT_RESOLUTION
) -> pd.DataFrame:
    """Convert a DataFrame containing ranges into a DataFrame containing intervals.

    Each interval is represented as a row in the DataFrame, with the start and end times as columns.

    Args:
        df (pd.DataFrame): A DataFrame containing ranges.
        threshold (int, optional): The threshold value used to determine the intervals.
            Defaults to DEFAULT_RESOLUTION.

    Returns:
        pd.DataFrame: A DataFrame containing intervals.
    """
    range_intervals_df = range_df_to_range_intervals_df(range_df, threshold)
    return range_intervals_df_to_intervals_df(range_intervals_df)


def identify_intervals(df, resolution, time_column="timestamp"):
    df = df.sort_values(time_column)
    ts = df[time_column]
    if is_datetime(ts):
        ts = ts.astype(int) // 1e9
    period_start = ts.diff() > resolution.total_seconds()
    df["interval_label"] = period_start.cumsum()
    return df


def intervals_merge_overlaps(df, threshold=timedelta(0), columns=None):
    """Merge overlapping time intervals.

    Considering a list of time-intervals at the input, merge the rows
    that have the same columns values + have some time overlaps.
    """

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

    return intervals


def intervals_concat_days(days_df, threshold=timedelta(0)):
    ret_dfs = []
    to_merge_left = None
    for day_df in tqdm(days_df):
        # merge the left over from the previous day

        if to_merge_left is not None:
            # we consider only beginning of the day
            to_merge_right = day_df.loc[
                # in daily, we consider only beginning of the day
                ((day_df["start"] - day_df["start"].dt.normalize()) <= threshold)
            ]

            # merge the intervals
            merged_df = intervals_merge_overlaps(
                pd.concat([to_merge_left, to_merge_right], ignore_index=True),
                threshold=threshold,
            )
            merged_df.drop("sub_intervals", axis=1, inplace=True)

            # replace the intervals in to_merge_right with the merged ones
            day_df = day_df.loc[day_df.index.difference(to_merge_right.index)]
            day_df = pd.concat([day_df, merged_df], ignore_index=True)

        # prepare the left over for the next day
        to_merge_left = day_df.loc[
            # end of day
            (
                (day_df["end"] - day_df["end"].dt.normalize())
                >= (timedelta(days=1) - threshold)
            )
            # with special case of end right at the midnight
            | ((day_df["end"] - day_df["end"].dt.normalize()) == timedelta(0))
        ]

        # add the rest of the day_df to the ret_dfs: no need to merge with anything
        day_df = day_df.loc[day_df.index.difference(to_merge_left.index)]
        ret_dfs.append(day_df)

    # finalizing the left over from the last day
    ret_dfs.append(to_merge_left)
    df = pd.concat(ret_dfs, ignore_index=True)
    return df


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


def one_hot_encode(df):
    index = df.columns[0]
    columns = df.columns[1]
    return df.assign(value=1).pivot_table(
        index=index, columns=columns, values="value", fill_value=0
    )
