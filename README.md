# Bayesian online changepoint detection

An implementation and benchmark of changepoint detection algorithms on synthetic time series, comparing online (BOCPD, particle filter) and offline (PELT, binary segmentation) approaches.

## What this does

Given a time series, the goal is to detect when the underlying process changed. This project implements and rigorously compares four algorithms:

- **BOCPD** — Bayesian Online Changepoint Detection. Processes data point-by-point, maintains a posterior over run length, outputs a changepoint probability at each step.
- **Particle filter** — online, tracks multiple weighted hypotheses about run length and regime parameters simultaneously.
- **PELT** — Pruned Exact Linear Time. Offline, finds the globally optimal segmentation via dynamic programming with pruning.
- **Binary segmentation** — offline, greedy baseline that recursively splits the series at the single best changepoint.

All four are evaluated against a synthetic benchmark with known ground truth, across varying noise levels, shift magnitudes, and changepoint counts.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Generate a benchmark dataset:

```bash
python changepoint_engine/generator.py
```

This saves a Parquet file to `data/benchmark.parquet` containing synthetic series with known changepoint locations, noise levels, and shift magnitudes stored as columns.

Run tests:

```bash
pytest tests/
```

## Background

- Killick, R., Fearnhead, P. and Eckley, I.A, [https://arxiv.org/pdf/1101.1438]
