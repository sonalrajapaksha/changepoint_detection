# implementation of binary segmentation
import numpy as np


def segment_cost(series):
    if len(series) == 0:
        return 0.0
    mean = np.mean(series)
    return np.sum((series - mean) ** 2)


def best_split(series, min_size=5):
    """
    find the split point that maximises reduction in cost
    """
    n = len(series)

    total_cost = segment_cost(series)

    best_gain = 0
    best_split = None

    for t in range(min_size, n - min_size):
        left = series[:t]
        right = series[t:]

        split_cost = segment_cost(left) + segment_cost(right)
        gain = total_cost - split_cost

        if gain > best_gain:
            best_gain = gain
            best_split = t

    return best_split, best_gain


def binary_segment(series, threshold, min_size=5, offset=0):
    """

    Args:
        series ():
        threshold ():
        min_size ():
        offset ():

    Returns:

    """
    if len(series) < 2 * min_size:
        return []

    split, gain = best_split(series, min_size)

    if split is None or gain < threshold:
        return []

    cp = offset + split

    left_cps = binary_segment(
        series[:split],
        threshold,
        min_size,
        offset,
    )

    right_cps = binary_segment(
        series[split:],
        threshold,
        min_size,
        offset + split,
    )

    return left_cps + [cp] + right_cps
