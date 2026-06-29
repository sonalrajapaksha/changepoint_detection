import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb


def detection_delay(
    true_cps: list[int], detected_cps: list[int], tolerance: int = 20
) -> float:
    """
    Average number of steps between each true changepoint and its closest
    detected changepoint, only counting true changepoints that were detected
    within the tolerance window.

    Returns nan if nothing was detected.
    """
    if not detected_cps:
        return np.nan

    delays = []
    for tcp in true_cps:
        dists = [abs(tcp - dcp) for dcp in detected_cps]
        min_dist = min(dists)
        if min_dist <= tolerance:
            delays.append(min_dist)

    return np.mean(delays) if delays else np.nan


def false_positive_rate(
    true_cps: list[int], detected_cps: list[int], tolerance: int = 20
) -> float:
    """
    Fraction of detected changepoints that are not within tolerance
    of any true changepoint.
    """
    if not detected_cps:
        return 0.0

    fp = 0
    for dcp in detected_cps:
        if not any(abs(dcp - tcp) <= tolerance for tcp in true_cps):
            fp += 1

    return fp / len(detected_cps)


def false_negative_rate(
    true_cps: list[int], detected_cps: list[int], tolerance: int = 20
) -> float:
    """
    Fraction of true changepoints that have no detected changepoint
    within tolerance.
    """
    if not true_cps:
        return 0.0

    fn = 0
    for tcp in true_cps:
        if not any(abs(tcp - dcp) <= tolerance for dcp in detected_cps):
            fn += 1

    return fn / len(true_cps)


def precision(
    true_cps: list[int], detected_cps: list[int], tolerance: int = 20
) -> float:
    """Fraction of detected changepoints that are true positives."""
    if not detected_cps:
        return 0.0
    return 1.0 - false_positive_rate(true_cps, detected_cps, tolerance)


def recall(true_cps: list[int], detected_cps: list[int], tolerance: int = 20) -> float:
    """Fraction of true changepoints that were detected."""
    if not true_cps:
        return 0.0
    return 1.0 - false_negative_rate(true_cps, detected_cps, tolerance)


def f1(true_cps: list[int], detected_cps: list[int], tolerance: int = 20) -> float:
    """
    Harmonic mean of precision and recall.
    Returns 0 if both precision and recall are 0.
    """
    p = precision(true_cps, detected_cps, tolerance)
    r = recall(true_cps, detected_cps, tolerance)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def evaluate_single(
    true_cps: list[int], detected_cps: list[int], tolerance: int = 20
) -> dict:
    """
    Compute all metrics for a single series.
    Returns a dict of metric name -> value.
    """
    return {
        "detection_delay": detection_delay(true_cps, detected_cps, tolerance),
        "false_positive_rate": false_positive_rate(true_cps, detected_cps, tolerance),
        "false_negative_rate": false_negative_rate(true_cps, detected_cps, tolerance),
        "precision": precision(true_cps, detected_cps, tolerance),
        "recall": recall(true_cps, detected_cps, tolerance),
        "f1": f1(true_cps, detected_cps, tolerance),
        "n_detected": len(detected_cps),
        "n_true": len(true_cps),
    }


def evaluate_all(results: list[dict]) -> dict:
    """
    Aggregate metrics across all series.
    Each entry in results is the output of evaluate_single for one series.
    Returns mean and std of each metric.
    """
    keys = [k for k in results[0].keys() if k not in ("n_detected", "n_true")]
    summary = {}
    for k in keys:
        vals = [r[k] for r in results if not np.isnan(r[k])]
        summary[f"{k}_mean"] = np.mean(vals) if vals else np.nan
        summary[f"{k}_std"] = np.std(vals) if vals else np.nan
    return summary


def save_results(results: list[dict], algorithm: str, path: str):
    """
    Save per-series results to Parquet with algorithm name as a column.
    """
    table = pa.table(
        {
            "algorithm": [algorithm] * len(results),
            "series_id": list(range(len(results))),
            "detection_delay": [r["detection_delay"] for r in results],
            "false_positive_rate": [r["false_positive_rate"] for r in results],
            "false_negative_rate": [r["false_negative_rate"] for r in results],
            "precision": [r["precision"] for r in results],
            "recall": [r["recall"] for r in results],
            "f1": [r["f1"] for r in results],
            "n_detected": [r["n_detected"] for r in results],
            "n_true": [r["n_true"] for r in results],
        }
    )
    pq.write_table(table, path)
    print(f"saved {len(results)} results to {path}")


def query_results(path: str, query: str):
    """
    Run a DuckDB SQL query on a saved results Parquet file.
    Useful for slicing results by algorithm, metric, or threshold.

    Example:
        query_results("data/results.parquet",
            "SELECT algorithm, AVG(f1) as mean_f1 GROUP BY algorithm ORDER BY mean_f1 DESC")
    """
    return duckdb.query(f"SELECT * FROM '{path}' WHERE {query}").df()


if __name__ == "__main__":
    from BS_PELT import pelt, binary_segmentation
    from cusum import CUSUM, detect as cusum_detect
    from SlidingTTest import SlidingTTest, detect as ttest_detect

    table = pq.read_table("data/benchmark.parquet")
    n_series = len(table)
    tolerance = 20

    algorithms = {
        "pelt": [],
        "binary_segmentation": [],
        "cusum": [],
        "sliding_ttest": [],
    }

    for i in range(n_series):
        series = np.array(table["series"][i].as_py())
        true_cps = table["changepoints"][i].as_py()

        # pelt
        detected = pelt(series, penalty=50.0)
        algorithms["pelt"].append(evaluate_single(true_cps, detected, tolerance))

        # binary segmentation
        detected = binary_segmentation(series, min_improvement=50.0)
        algorithms["binary_segmentation"].append(
            evaluate_single(true_cps, detected, tolerance)
        )

        # cusum
        cusum = CUSUM(
            target_mean=np.mean(series[:20]),
            delta=2.0,
            sigma=np.std(series[:20]),
            threshold=10.0,
        )
        cp_flags, _, _ = cusum.run(series)
        detected = cusum_detect(cp_flags, min_gap=30)
        algorithms["cusum"].append(evaluate_single(true_cps, detected, tolerance))

        # sliding t-test
        ttest = SlidingTTest(window_size=25, threshold=4.0)
        cp_flags, _ = ttest.run(series)
        detected = ttest_detect(cp_flags, min_gap=30)
        algorithms["sliding_ttest"].append(
            evaluate_single(true_cps, detected, tolerance)
        )

    # print summary table
    print(f"\n{'algorithm':<25} {'f1':>8} {'delay':>8} {'fpr':>8} {'recall':>8}")
    print("-" * 60)
    for algo, results in algorithms.items():
        summary = evaluate_all(results)
        print(
            f"{algo:<25}"
            f"{summary['f1_mean']:>8.3f}"
            f"{summary['detection_delay_mean']:>8.1f}"
            f"{summary['false_positive_rate_mean']:>8.3f}"
            f"{summary['recall_mean']:>8.3f}"
        )

    # save each algorithm's results to parquet
    for algo, results in algorithms.items():
        save_results(results, algo, f"data/results_{algo}.parquet")

    print("\nresults saved to data/")
