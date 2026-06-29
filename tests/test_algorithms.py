import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "changepoint_engine"))

from BS_PELT import pelt, binary_segmentation, segment_cost
from cusum import CUSUM, detect as cusum_detect
from SlidingTTest import SlidingTTest, detect as ttest_detect
from metrics import f1, precision, recall, detection_delay, false_positive_rate


# shared fixture — obvious single changepoint
@pytest.fixture
def simple_series():
    rng = np.random.default_rng(42)
    return np.concatenate([rng.normal(0.0, 1.0, 50), rng.normal(10.0, 1.0, 50)]), [50]


@pytest.fixture
def flat_series():
    return np.random.default_rng(0).normal(5.0, 1.0, 200), []


# pelt
def test_pelt_detects_changepoint(simple_series):
    series, true_cps = simple_series
    detected = pelt(series, penalty=10.0)
    assert any(abs(d - true_cps[0]) <= 5 for d in detected)


def test_pelt_no_changepoint_flat(flat_series):
    series, _ = flat_series
    assert len(pelt(series, penalty=50.0)) <= 1


def test_pelt_sorted(simple_series):
    series, _ = simple_series
    detected = pelt(series, penalty=10.0)
    assert detected == sorted(detected)


def test_pelt_higher_penalty_fewer_cps(simple_series):
    series, _ = simple_series
    assert len(pelt(series, penalty=5.0)) >= len(pelt(series, penalty=100.0))


# binary segmentation
def test_binseg_detects_changepoint(simple_series):
    series, true_cps = simple_series
    detected = binary_segmentation(series, min_improvement=5.0)
    assert any(abs(d - true_cps[0]) <= 5 for d in detected)


def test_binseg_no_changepoint_flat(flat_series):
    series, _ = flat_series
    assert len(binary_segmentation(series, min_improvement=50.0)) == 0


def test_binseg_sorted(simple_series):
    series, _ = simple_series
    detected = binary_segmentation(series, min_improvement=5.0)
    assert detected == sorted(detected)


# cusum
def test_cusum_detects_changepoint(simple_series):
    series, true_cps = simple_series
    cusum = CUSUM(target_mean=0.0, delta=2.0, sigma=1.0, threshold=5.0)
    cp_flags, _, _ = cusum.run(series)
    detected = cusum_detect(cp_flags)
    assert any(abs(d - true_cps[0]) <= 15 for d in detected)


def test_cusum_no_detection_flat(flat_series):
    series, _ = flat_series
    cusum = CUSUM(target_mean=5.0, delta=2.0, sigma=1.0, threshold=10.0)
    cp_flags, _, _ = cusum.run(series)
    assert len(cusum_detect(cp_flags)) == 0


def test_cusum_stats_non_negative(simple_series):
    series, _ = simple_series
    cusum = CUSUM(target_mean=0.0, delta=2.0, sigma=1.0, threshold=5.0)
    _, s_up, s_down = cusum.run(series)
    assert np.all(s_up >= 0) and np.all(s_down >= 0)


# sliding t-test
def test_ttest_detects_changepoint(simple_series):
    series, true_cps = simple_series
    detector = SlidingTTest(window_size=20, threshold=3.0)
    cp_flags, _ = detector.run(series)
    detected = ttest_detect(cp_flags, min_gap=20)
    assert any(abs(d - true_cps[0]) <= 20 for d in detected)


def test_ttest_no_detection_flat(flat_series):
    series, _ = flat_series
    detector = SlidingTTest(window_size=20, threshold=5.0)
    cp_flags, _ = detector.run(series)
    assert len(ttest_detect(cp_flags)) == 0


def test_ttest_same_result_twice(simple_series):
    series, _ = simple_series
    detector = SlidingTTest(window_size=20, threshold=3.0)
    flags1, stats1 = detector.run(series)
    flags2, stats2 = detector.run(series)
    np.testing.assert_array_equal(flags1, flags2)


# metrics
def test_metrics_perfect():
    assert f1([50, 150], [50, 150]) == 1.0
    assert precision([50, 150], [50, 150]) == 1.0
    assert recall([50, 150], [50, 150]) == 1.0
    assert detection_delay([50, 150], [50, 150]) == 0.0


def test_metrics_no_detection():
    assert f1([50, 150], []) == 0.0
    assert recall([50, 150], []) == 0.0
    assert np.isnan(detection_delay([50, 150], []))


def test_metrics_all_false_positives():
    assert false_positive_rate([50], [10], tolerance=20) == 1.0


def test_metrics_within_tolerance():
    assert recall([50], [60], tolerance=20) == 1.0
    assert detection_delay([50], [60], tolerance=20) == 10.0
