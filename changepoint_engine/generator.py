import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from dataclasses import dataclass


@dataclass
class Segment:
    start: int
    end: int
    mean: float
    std: float


class SeriesGenerator:
    def __init__(self, n_series: int = 100, length: int = 500, seed: int = 42):
        self.n_series = n_series
        self.length = length
        self.rng = np.random.default_rng(seed)

    def generate_one(
        self, n_changepoints: int = 3, noise: float = 1.0, shift: float = 3.0
    ):
        changepoint_locations = sorted(
            self.rng.choice(self.length - 2, n_changepoints, replace=False) + 1
        )
        breakpoints = [0] + changepoint_locations + [self.length]

        series = np.zeros(self.length)
        segments = []
        current_mean = 0.0

        for i in range(len(breakpoints) - 1):
            start, end = breakpoints[i], breakpoints[i + 1]
            current_mean += self.rng.choice([-1, 1]) * shift
            segment_data = self.rng.normal(current_mean, noise, end - start)
            series[start:end] = segment_data
            segments.append(Segment(start, end, current_mean, noise))

        return series, segments

    def generate(self, n_changepoints: int = 3, noise: float = 1.0, shift: float = 3.0):
        all_series = []
        all_changepoints = []
        for i in range(self.n_series):
            series, segments = self._generate_one(n_changepoints, noise, shift)
            all_series.append(series)
            all_changepoints.append(segments)
        return all_series, all_changepoints

    def save(
        self, path: str, n_changepoints: int = 3, noise: float = 1.0, shift: float = 3.0
    ):
        series_list, cp_list = self.generate(n_changepoints, noise, shift)

        table = pa.table(
            {
                "id": list(range(self.n_series)),
                "series": pa.array(series_list, type=pa.list_(pa.float64())),
                "changepoints": pa.array(cp_list, type=pa.list_(pa.int64())),
                "n_changepoints": [n_changepoints] * self.n_series,
                "noise": [noise] * self.n_series,
                "shift": [shift] * self.n_series,
            }
        )

        pq.write_table(table, path)
        print(f"saved {self.n_series} series to {path}")
