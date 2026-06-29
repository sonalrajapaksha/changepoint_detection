import numpy as np


class CUSUM:
    def __init__(
        self,
        target_mean: float = 0.0,
        delta: float = 1.0,
        sigma: float = 1.0,
        threshold: float = 5.0,
    ):
        self.target_mean = target_mean
        self.delta = delta
        self.sigma = sigma
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.s_up = 0.0  # upward CUSUM statistic
        self.s_down = 0.0  # downward CUSUM statistic

    def update(self, x: float) -> bool:
        """
        Process a single observation.

        Returns True if a changepoint is detected, False otherwise.
        Resets the statistics after detection.
        """
        z = (x - self.target_mean) / self.sigma

        self.s_up = max(0.0, self.s_up + z - self.delta / 2)
        self.s_down = max(0.0, self.s_down - z - self.delta / 2)

        if self.s_up > self.threshold or self.s_down > self.threshold:
            self.target_mean = x
            self.reset()
            return True

        return False

    def run(self, series: np.ndarray) -> tuple:
        T = len(series)
        cp_flags = np.zeros(T, dtype=bool)
        s_up = np.zeros(T)
        s_down = np.zeros(T)

        self.reset()

        for t, x in enumerate(series):
            cp_flags[t] = self.update(x)
            s_up[t] = self.s_up
            s_down[t] = self.s_down

        return cp_flags, s_up, s_down


def detect(cp_flags: np.ndarray, min_gap: int = 20) -> list[int]:
    cps = []
    for i, flag in enumerate(cp_flags):
        if not flag:
            continue
        if not cps or i - cps[-1] >= min_gap:
            cps.append(i)
    return cps


if __name__ == "__main__":
    import pyarrow.parquet as pq
    import matplotlib.pyplot as plt

    table = pq.read_table("data/benchmark.parquet")
    series = np.array(table["series"][0].as_py())
    true_cps = table["changepoints"][0].as_py()

    # set target mean and sigma from first segment
    target_mean = np.mean(series[:40])
    sigma = np.std(series[:40])

    cusum = CUSUM(
        target_mean=target_mean,
        delta=2.0,  # detect shifts of 2 sigma
        sigma=sigma,
        threshold=15.0,
    )

    cp_flags, s_up, s_down = cusum.run(series)

    print("true changepoints:", true_cps)
    print("detected:", detect(cp_flags))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    axes[0].plot(series, color="steelblue", linewidth=0.8)
    for cp in true_cps:
        axes[0].axvline(x=cp, color="red", linestyle="--")
    axes[0].set_title("series with true changepoints")

    axes[1].plot(s_up, color="green", linewidth=0.8, label="S+ (upward)")
    axes[1].plot(s_down, color="orange", linewidth=0.8, label="S- (downward)")
    axes[1].axhline(
        y=cusum.threshold,
        color="black",
        linestyle=":",
        linewidth=0.8,
        label="threshold",
    )
    for cp in true_cps:
        axes[1].axvline(x=cp, color="red", linestyle="--")
    axes[1].set_title("CUSUM statistics")
    axes[1].legend()

    detected = detect(cp_flags)
    axes[2].plot(series, color="steelblue", linewidth=0.8, alpha=0.5)
    for cp in true_cps:
        axes[2].axvline(
            x=cp, color="red", linestyle="--", label="true" if cp == true_cps[0] else ""
        )
    for cp in detected:
        axes[2].axvline(
            x=cp,
            color="green",
            linestyle="-",
            linewidth=1.2,
            label="detected" if cp == detected[0] else "",
        )
    axes[2].set_title("detections vs true changepoints")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig("data/cusum_output.png")
    plt.show()
