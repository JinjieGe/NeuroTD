"""Discrete-window delay estimator matching NeuroTD (rectangular, continuous=False)."""
from __future__ import annotations
from revision_runtime import resolve_path

import numpy as np


def estimate_pair(x, y, W=1001, H=500, max_shift=500):
    """Return centers, signed shifts (−best_tau), mse at tau=0, mse at optimum.

    Tie-break: smallest tau (most negative) wins, as in the original loop.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    N = x.size
    starts = np.arange(max_shift, N - W - max_shift + 1, H, dtype=np.int64)
    nwin = starts.size
    min_mse = np.full(nwin, np.inf)
    best_tau = np.zeros(nwin, dtype=np.int32)
    mse0 = np.full(nwin, np.nan)
    c0 = np.concatenate([[0.0], np.cumsum(x * x)])
    # tau loop: prefix sums of (x[t]-y[t-tau])^2
    for tau in range(-max_shift, max_shift + 1):
        t0 = max(0, tau)
        t1 = min(N, N + tau)
        d2 = np.zeros(N)
        d2[t0:t1] = (x[t0:t1] - y[t0 - tau:t1 - tau]) ** 2
        c = np.concatenate([[0.0], np.cumsum(d2)])
        sums = c[starts + W] - c[starts]
        valid = (starts - tau >= 0) & (starts - tau + W <= N)
        mse = np.where(valid, sums / W, np.inf)
        if tau == 0:
            mse0 = sums / W
        better = mse < min_mse
        min_mse[better] = mse[better]
        best_tau[better] = tau
    centers = starts + (W - 1) // 2
    return centers, (-best_tau).astype(np.int16), mse0.astype(np.float32), min_mse.astype(np.float32)


def verify_against_original(x, y, W=1001, H=500):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(resolve_path('repo', 'src'))))
    from neurotd.time_varying_shift import compute_windowed_mse_with_hop
    c1, s1, _, _ = estimate_pair(x, y, W, H, W // 2)
    c0, s0 = compute_windowed_mse_with_hop(x, y, W, H, "rectangle", False)
    return np.array_equal(c1, np.asarray(c0, dtype=int)), np.array_equal(s1, np.asarray(s0, dtype=np.int16))
