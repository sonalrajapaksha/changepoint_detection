# Changepoint Detection Engine

A from-scratch implementation and empirical benchmark of four changepoint detection algorithms on synthetic time series. Compares offline methods (PELT, binary segmentation) against online methods (CUSUM, sliding t-test) across detection accuracy, lag, and false positive rate.

---

## Algorithms

**Offline**

- **PELT** — finds the globally optimal segmentation by minimising a penalised cost function via dynamic programming with pruning. Exact and O(n) in the number of changepoints.
- **Binary segmentation** — greedy baseline that recursively splits the series at the single best changepoint. Not globally optimal but fast and interpretable.

**Online**

- **CUSUM** — accumulates evidence of a mean shift via two running statistics (upward and downward). Flags a changepoint when either exceeds a threshold and resets. O(1) per update.
- **Sliding t-test** — maintains two adjacent windows of equal size and computes Welch's t-statistic between them at each step using O(1) running sum-of-squares accumulators. Flags a changepoint when the statistic exceeds a threshold.

---

## Results

Evaluated on 1000 synthetic series of length 500 with 3 changepoints each, shift magnitude ~3 units, within-segment noise std ~1.5. Tolerance window of 20 steps.

```
algorithm                      f1    delay   fpr    recall
------------------------------------------------------------
pelt                        0.962     0.3   0.000   0.947
binary_segmentation         0.937     0.4   0.000   0.913
cusum                       0.846     5.0   0.161   0.897
sliding_ttest               0.948    11.1   0.007   0.930
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Generate benchmark data:

```bash
python changepoint_engine/generator.py
```

Run any detector:

```bash
python changepoint_engine/pelt.py
python changepoint_engine/particle.py
python changepoint_engine/cusum.py
python changepoint_engine/sliding_ttest.py
```

Run full benchmark and save results:

```bash
python changepoint_engine/metrics.py
```

Query results with DuckDB:

```python
from metrics import query_results
query_results("data/results_pelt.parquet", "f1 < 0.5")
```

Run tests:

```bash
pytest tests/
```
