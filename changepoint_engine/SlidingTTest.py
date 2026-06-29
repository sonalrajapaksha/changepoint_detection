from collections import deque
import numpy as np
import pyarrow.parquet as pq
import matplotlib.pyplot as plt


class SlidingTTest:
    def __init__(self, window_size: int = 25, threshold: float = 4.0):
        self.W = window_size
        self.threshold = threshold
        self.ref_window = deque(maxlen=window_size)
        self.test_window = deque(maxlen=window_size)
        self.sum_ref = 0.0
        self.ss_ref = 0.0
        self.sum_test = 0.0
        self.ss_test = 0.0

    def reset(self):
        self.ref_window.clear()
        self.test_window.clear()
        self.sum_ref = 0.0
        self.ss_ref = 0.0
        self.sum_test = 0.0
        self.ss_test = 0.0

    def update(self, x: float) -> tuple[bool, float]:
        # filling test window
        if len(self.test_window) < self.W:
            self.test_window.append(x)
            self.sum_test += x
            self.ss_test += x * x
            return False, 0.0

        # priming reference window
        if len(self.ref_window) < self.W:
            x_mid = self.test_window.popleft()
            self.sum_test -= x_mid
            self.ss_test -= x_mid * x_mid
            self.ref_window.append(x_mid)
            self.sum_ref += x_mid
            self.ss_ref += x_mid * x_mid
            self.test_window.append(x)
            self.sum_test += x
            self.ss_test += x * x
            if len(self.ref_window) < self.W:
                return False, 0.0

        # both windows full — O(1) shift
        else:
            x_drop = self.ref_window.popleft()
            self.sum_ref -= x_drop
            self.ss_ref -= x_drop * x_drop
            x_mid = self.test_window.popleft()
            self.sum_test -= x_mid
            self.ss_test -= x_mid * x_mid
            self.ref_window.append(x_mid)
            self.sum_ref += x_mid
            self.ss_ref += x_mid * x_mid
            self.test_window.append(x)
            self.sum_test += x
            self.ss_test += x * x

        mu_ref = self.sum_ref / self.W
        mu_test = self.sum_test / self.W
        var_ref = (self.ss_ref - (self.sum_ref**2) / self.W) / (self.W - 1) + 1e-6
        var_test = (self.ss_test - (self.sum_test**2) / self.W) / (self.W - 1) + 1e-6
        t_stat = abs(mu_test - mu_ref) / np.sqrt(
            (var_ref / self.W) + (var_test / self.W)
        )

        return t_stat > self.threshold, t_stat

    def run(self, series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.reset()
        n = len(series)
        cp_flags = np.zeros(n, dtype=bool)
        t_stats = np.zeros(n)

        for i, x in enumerate(series):
            is_cp, score = self.update(x)
            t_stats[i] = score
            if is_cp:
                cp_flags[i - self.W] = True

        return cp_flags, t_stats


def detect(cp_flags: np.ndarray, min_gap: int = 20) -> list[int]:
    cps = []
    for i, flag in enumerate(cp_flags):
        if not flag:
            continue
        if not cps or i - cps[-1] >= min_gap:
            cps.append(i)
    return cps


if __name__ == "__main__":
    table = pq.read_table("data/benchmark.parquet")
    series = np.array(table["series"][0].as_py())
    true_cps = table["changepoints"][0].as_py()

    detector = SlidingTTest(window_size=25, threshold=4.0)
    cp_flags, t_stats = detector.run(series)

    print("true changepoints:", true_cps)
    print("detected:", detect(cp_flags, min_gap=30))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))

    axes[0].plot(series, color="steelblue", linewidth=0.8)
    for cp in true_cps:
        axes[0].axvline(x=cp, color="red", linestyle="--")
    axes[0].set_title("series with true changepoints")

    axes[1].plot(t_stats, color="darkorange", linewidth=0.8)
    axes[1].axhline(
        y=detector.threshold,
        color="black",
        linestyle=":",
        linewidth=0.8,
        label="threshold",
    )
    for cp in true_cps:
        axes[1].axvline(x=cp, color="red", linestyle="--")
    axes[1].set_title("Welch t-statistic")
    axes[1].set_ylabel("t-stat")
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
    plt.savefig("data/ttest_output.png")
    plt.show()
