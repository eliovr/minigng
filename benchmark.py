"""Micro-benchmark for MiniGNG.fit.

Times `fit` on synthetic Gaussian-blob data. The work is dominated by the
per-signal nearest-unit search in `partial_fit`, so this exercises exactly the
hot loop touched by the prototype-matrix / squared-distance optimizations.

Run with the current working tree to get one number; `git stash` the library
change and re-run to get the baseline.

Usage:
    python benchmark.py
"""

import time
import numpy as np

from minigng import MiniGNG


def make_blobs(n_samples, dim, n_centers, seed=0):
    rng = np.random.RandomState(seed)
    centers = rng.uniform(-10, 10, size=(n_centers, dim))
    assignments = rng.randint(0, n_centers, size=n_samples)
    X = centers[assignments] + rng.normal(0, 1.0, size=(n_samples, dim))
    return X.astype(np.float32)


def bench(name, X, repeats=3, **params):
    fit_times, pred_times = [], []
    for _ in range(repeats):
        np.random.seed(0)  # fix sampling/shuffling so each run is comparable
        gng = MiniGNG(**params)
        t0 = time.perf_counter()
        gng.fit(X)
        fit_times.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        gng.predict(X)
        pred_times.append(time.perf_counter() - t0)
    fit_best, pred_best = min(fit_times), min(pred_times)
    n_signals = params["n_epochs"] * len(X)
    print(
        f"{name:<28} fit={fit_best:7.3f}s  "
        f"({n_signals / fit_best:>9,.0f} sig/s)  "
        f"predict={pred_best * 1e3:7.2f}ms  units={len(gng.units):3d}"
    )
    return fit_best


if __name__ == "__main__":
    print("minigng fit benchmark (best of 3)\n")

    configs = [
        ("small  (N=1k d=8 u=100)",
         dict(n=1000, dim=8, centers=8),
         dict(max_units=100, n_epochs=10)),
        ("medium (N=3k d=16 u=300)",
         dict(n=3000, dim=16, centers=12),
         dict(max_units=300, n_epochs=10)),
        ("large  (N=5k d=32 u=500)",
         dict(n=5000, dim=32, centers=16),
         dict(max_units=500, n_epochs=8)),
        ("untangle (N=3k d=16 u=300)",
         dict(n=3000, dim=16, centers=12),
         dict(max_units=300, n_epochs=10, untangle=True, max_size_connect=3)),
    ]

    for name, data_kw, params in configs:
        X = make_blobs(data_kw["n"], data_kw["dim"], data_kw["centers"])
        bench(name, X, **params)
