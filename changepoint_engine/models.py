import numpy as np
from scipy.stats import norm, t as student_t


class GaussianModel:
    """
    updates running mean and variance using Welford's online algorithm
    for a gaussian distribution

    used for both BOCPD and particle filter as they are online algorithms
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # sets to initial state
        self.mean = 0.0
        self.var = 1.0
        self.n = 0
        self._M2 = 0.0

    def update(self, x: float):
        # incorporate new observation
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._M2 += delta * delta2
        self.var = max(1e-8, self._M2 / self.n)

    def predictive_log_prob(self, x: float) -> float:
        return norm.logpdf(x, loc=self.mean, scale=np.sqrt(self.var))


class StudentTModel:
    """
    identical to gaussian model, but uses student-t distribution
    used for its heavier tails which makes it less sensitive to outliers
    """

    def __init__(self, df: float = 3.0):
        self.df = df
        self.reset()

    def reset(self):
        self.mean = 0.0
        self.var = 1.0
        self.n = 0
        self._M2 = 0.0

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self._M2 += delta * delta2
        self.var = max(1e-8, self._M2 / self.n)

    def predictive_log_prob(self, x: float) -> float:
        return student_t.logpdf(x, df=self.df, loc=self.mean, scale=np.sqrt(self.var))


if __name__ == "__main__":
    import pyarrow.parquet as pq

    table = pq.read_table("data/benchmark.parquet")
    series = np.array(table["series"][0].as_py())

    g = GaussianModel()
    s = StudentTModel(df=3.0)

    for x in series[:44]:
        g.update(x)
        s.update(x)

    normal_obs = series[10]
    cp_obs = series[45]

    print(f"gaussian  - normal point: {g.predictive_log_prob(normal_obs):.3f}")
    print(f"gaussian  - changepoint:  {g.predictive_log_prob(cp_obs):.3f}")
    print(f"student-t - normal point: {s.predictive_log_prob(normal_obs):.3f}")
    print(f"student-t - changepoint:  {s.predictive_log_prob(cp_obs):.3f}")
