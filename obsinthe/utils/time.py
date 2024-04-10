from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from functools import reduce
from typing import Dict

import numpy as np
import pandas as pd
from dateutil.rrule import DAILY
from dateutil.rrule import rrule


def normalize_tz(time: datetime):
    """Ensure time is normalized to UTC"""
    if time.tzinfo is None:
        return time.replace(tzinfo=timezone.utc)
    return time.astimezone(timezone.utc)


def datetime_start_of_day(dt: datetime) -> datetime:
    return datetime.combine(dt.date(), datetime.min.time(), tzinfo=timezone.utc)


def gen_daily_intervals(start, end):
    start, end = normalize_tz(start), normalize_tz(end)
    ticks = [
        d.replace(tzinfo=timezone.utc)
        for d in rrule(DAILY, dtstart=start.date(), until=end.date())
    ]
    ints = list(zip(ticks, ticks[1:]))
    last_day = ticks[-1]
    if datetime.combine(last_day, datetime.min.time(), tzinfo=timezone.utc) < end:
        ints.append((last_day, last_day + timedelta(days=1)))

    return ints


def intervals_daily_split(df: pd.DataFrame) -> Dict[datetime, pd.DataFrame]:
    """Split an dataframe with intervals into individual days.

    Args:
        df (pandas.DataFrame): A DataFrame with intervals.
    """
    df.set_index("digest")
    daily_alerts_digests_intervals = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        for interval in row["intervals"]:
            interval = [
                datetime.utcfromtimestamp(d).replace(tzinfo=timezone.utc)
                for d in interval
            ]
            interval_start, interval_end = interval

            for start_day, end_day in gen_daily_intervals(interval_start, interval_end):
                start = max(start_day, interval_start)
                end = min(end_day, interval_end)
                daily_alerts_digests_intervals[start_day][row["digest"]].append(
                    (start.timestamp(), end.timestamp())
                )

    ret = {}
    df_base = df.drop(columns=["intervals"])

    for day, digests in daily_alerts_digests_intervals.items():
        day_df = pd.DataFrame(
            data={"digest": digests.keys(), "intervals": digests.values()}
        )
        ret[day] = df_base.merge(day_df, on="digest")

    return ret


def merge_daily_split_intervals(day_dfs):
    day_dfs = [df.set_index("digest") for df in day_dfs if df is not None]

    def merge_row_intervals(row):
        left, right = row["left"], row["right"]
        if isinstance(left, float) and pd.isnull(left):
            return right
        if isinstance(right, float) and pd.isnull(right):
            return left

        # ensure we have a list of lists
        left = [i.tolist() if i is np.array else list(i) for i in left]
        right = [i.tolist() if i is np.array else list(i) for i in right]

        if left[-1][1] == right[0][0]:
            left[-1][1] = right[0][1]

            right = right[1:]
        return [np.array(i) for i in (left + right)]

    def merge_adjacent_days(day_one, day_two):
        df = pd.DataFrame({"left": day_one, "right": day_two})
        ret = df.apply(merge_row_intervals, axis=1)
        return ret

    base = reduce(
        lambda a, b: a.combine_first(b),
        [df.drop(columns=["intervals"]) for df in day_dfs],
    )

    intervals = reduce(merge_adjacent_days, [d["intervals"] for d in day_dfs])
    intervals.name = "intervals"

    return base.join(intervals)
