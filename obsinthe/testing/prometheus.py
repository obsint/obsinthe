import abc
import json
import math
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from typing import Callable
from typing import List
from typing import SupportsFloat
from typing import Tuple
from typing import Union


DEFAULT_START_TIME = datetime(2024, 1, 1, tzinfo=timezone.utc)
DEFAULT_END_TIME = datetime(2024, 1, 3, tzinfo=timezone.utc)
DEFAULT_RESOLUTION = timedelta(minutes=1)


# The TimeSpec can be defined as:
# - a datetime object, representing a specific point in time
# - a timedelta object, representing delta from the start of the dataset
# - None, representing the start or end of the dataset (depending on the context)
TimeSpec = Union[datetime, timedelta, None]

# The ValueSpec type is a union of a float and a callable.
# The callable is a function that takes a single argument, the timestamp, and
# returns a float: a corresponding value for the timestamp.
ValueSpec = Union[SupportsFloat, Callable[[datetime], SupportsFloat]]


@dataclass
class Interval:
    start: TimeSpec
    end: TimeSpec
    value: ValueSpec

    def normalize_start_end(
        self, dataset_start: datetime, dataset_end: datetime
    ) -> Tuple[datetime, datetime]:
        """
        Normalize the start and end times of the interval.

        Turns the TimeSpec for `start` and `end` into a datetime object.
        """
        if isinstance(self.start, timedelta):
            start = dataset_start + self.start
        elif self.start is None:
            start = dataset_start
        else:
            start = self.start

        start = max(start, dataset_start)
        start = normalize_tz(start)

        if isinstance(self.end, timedelta):
            end = dataset_start + self.end
        elif self.end is None:
            end = dataset_end
        else:
            end = self.end

        end = min(end, dataset_end)
        end = normalize_tz(end)

        return start, end

    def eval(self, timestamp: datetime) -> float:
        """
        Evaluate the value for the given timestamp.
        """
        if isinstance(self.value, SupportsFloat):
            return float(self.value)
        return float(self.value(timestamp))


class TimeSeriesBuilder:
    def __init__(self, labels: dict[str, str]):
        self.labels = labels
        self.intervals: List[Interval] = []

    def sample(self, timestamp: datetime, value: ValueSpec):
        """
        Add a single sample to the time series.

        The timestamp needs to match the resolution of the underlying dataset.

        See `interval` for more details on TimeSpec
        """
        self.intervals.append(Interval(timestamp, timestamp, value))
        return self

    def interval(self, start: TimeSpec, end: TimeSpec, value: ValueSpec):
        """
        Define value for a given interval.

        The samples will be evaluated depending on the resolution of the
        underlying dataset.
        """
        self.intervals.append(Interval(start, end, value))
        return self

    def value(self, value: ValueSpec):
        """
        Define value for the entire time series.

        The value will be evaluated depending on the resolution of the
        underlying dataset.
        """
        self.intervals.append(Interval(None, None, value))
        return self

    def build(
        self, start: datetime, end: datetime, step: timedelta
    ) -> List[Tuple[int, float]]:
        start, end = normalize_tz(start), normalize_tz(end)

        ret = []
        for interval in self.intervals:
            int_start, int_end = interval.normalize_start_end(start, end)

            current = next_sample(start, step, int_start)

            while current <= int_end:
                ret.append((int(current.timestamp()), interval.eval(current)))
                current += step

        return ret


class PromDatasetBuilderBase(abc.ABC):
    """Abstract base class for Prometheus dataset builders."""

    def __init__(
        self,
    ):
        self.time_series: List[TimeSeriesBuilder] = []

    def ts(self, labels: dict[str, str]) -> TimeSeriesBuilder:
        """
        Create a new time series with the given labels.
        """
        ts = TimeSeriesBuilder(labels)
        self.time_series.append(ts)
        return ts

    @abc.abstractclassmethod
    def build_raw(self) -> List[dict]:
        """
        Build the dataset as returned by Prometheus after parsing JSON.
        """
        pass

    def build_json(self) -> str:
        """
        Build the dataset as a JSON string.
        """

        return json.dumps(self.build_raw())


class PromInstantDatasetBuilder(PromDatasetBuilderBase):
    """
    A builder for Prometheus instant query datasets.

    Generates a dataset for a single point in time.
    """

    def __init__(
        self,
        time=DEFAULT_START_TIME,
    ):
        super().__init__()
        self.time = normalize_tz(time)

    def build_raw(self) -> List[dict]:
        ret = []

        for ts in self.time_series:
            # We don't need DEFAULT_RESOLUTION here, as we only have a single point,
            # adding it just to be consistent with the other builders.
            values = ts.build(self.time, self.time, DEFAULT_RESOLUTION)
            if values:
                value = values[0]
                value = [value[0], str(value[1])]
                ret.append({"metric": ts.labels, "value": value})

        return ret


class PromRangeDatasetBuilder(PromDatasetBuilderBase):
    """
    A builder for Prometheus range query datasets.
    """

    def __init__(
        self,
        start_time=DEFAULT_START_TIME,
        end_time=DEFAULT_END_TIME,
        resolution=DEFAULT_RESOLUTION,
    ):
        super().__init__()
        self.start_time = normalize_tz(start_time)
        self.end_time = normalize_tz(end_time)
        self.resolution = resolution

    def build_raw(self) -> List[dict]:
        ret = []

        for ts in self.time_series:
            values = ts.build(self.start_time, self.end_time, self.resolution)
            values = [[t, str(v)] for t, v in values]
            ret.append({"metric": ts.labels, "values": values})

        return ret


def next_sample(start: datetime, resolution: timedelta, current: datetime):
    """
    Return the next sample time after the current time.
    """
    steps = math.ceil((current - start) / resolution)
    return start + steps * resolution


def normalize_tz(time: datetime):
    """Ensure time is normalized to UTC"""
    if time.tzinfo is None:
        return time.replace(tzinfo=timezone.utc)
    return time.astimezone(timezone.utc)
