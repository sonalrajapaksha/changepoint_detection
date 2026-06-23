import numpy as np


def segment_cost(series: np.ndarray, start: int, end: int) -> float:
    segment = series[start:end]
    n = len(segment)
    if n == 0:
        return 0.0
    var = np.var(segment)
    if var < 1e-8:
        var = 1e-8
    return n * np.log(var)


def pelt(series: np.ndarray, penalty: float = None) -> list[int]:
    n = len(series)
    if penalty is None:
        penalty = 2 * np.log(n)

    F = np.zeros(n + 1)
    F[0] = -penalty
    last_cp = np.zeros(n + 1, dtype=int)
    candidates = [0]

    for t in range(1, n + 1):
        costs = [F[s] + segment_cost(series, s, t) + penalty for s in candidates]
        best_idx = int(np.argmin(costs))
        F[t] = costs[best_idx]
        last_cp[t] = candidates[best_idx]

        candidates = [s for s, c in zip(candidates, costs) if c <= F[t] + penalty]
        candidates.append(t)

    # backtrack to recover changepoints
    changepoints = []
    t = n
    while t > 0:
        s = last_cp[t]
        if s != 0:
            changepoints.append(s)
        t = s

    return sorted(changepoints)


def binary_segmentation(
    series: np.ndarray, min_improvement: float = 5.0, max_changepoints: int = 10
) -> list[int]:
    changepoints = []
    segments_to_check = [(0, len(series))]

    while segments_to_check and len(changepoints) < max_changepoints:
        start, end = segments_to_check.pop(0)
        if end - start < 4:
            continue

        baseline_cost = segment_cost(series, start, end)
        best_split = None
        best_cost = baseline_cost

        for t in range(start + 2, end - 2):
            cost = segment_cost(series, start, t) + segment_cost(series, t, end)
            if cost < best_cost:
                best_cost = cost
                best_split = t

        if best_split is not None and (baseline_cost - best_cost) > min_improvement:
            changepoints.append(best_split)
            segments_to_check.append((start, best_split))
            segments_to_check.append((best_split, end))

    return sorted(changepoints)


if __name__ == "__main__":
    import pyarrow.parquet as pq

    table = pq.read_table("data/benchmark.parquet")
    series = np.array(table["series"][0].as_py())
    true_cps = table["changepoints"][0].as_py()

    print("true changepoints:  ", true_cps)
    print("pelt detected:      ", pelt(series, penalty=50.0))
    print("binseg detected:    ", binary_segmentation(series, min_improvement=50.0))
