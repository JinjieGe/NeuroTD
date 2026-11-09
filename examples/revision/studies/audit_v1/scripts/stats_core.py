"""Trial-level stratified circular block bootstrap for the Figure 4c region comparison.

All four regions are evaluated from one resampling-count matrix ``C`` (resamples x trials),
so a drawn trial contributes every region, channel pair and window it contains.
Blocks are drawn inside one session only and wrap from the last trial of that session
to its first trial.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

REGIONS = ["Amygdala", "Hippocampus", "Middle Temporal Gyrus", "Superior Temporal Gyrus"]
REGION_SHORT = {"Amygdala": "AMY", "Hippocampus": "HIP",
                "Middle Temporal Gyrus": "MTG", "Superior Temporal Gyrus": "STG"}
COMPARISONS = list(combinations(range(len(REGIONS)), 2))  # (a, b): difference = region a - region b
N_COMP = len(COMPARISONS)


# ---------------------------------------------------------------------------- resampling
def circular_block_positions(rng: np.random.Generator, n: int, L: int, b: int) -> np.ndarray:
    """(b, n) positions 0..n-1 from ceil(n/L) circular blocks of length L, truncated to n."""
    nb = -(-n // L)
    starts = rng.integers(0, n, size=(b, nb))
    pos = (starts[:, :, None] + np.arange(L)[None, None, :]) % n
    return pos.reshape(b, nb * L)[:, :n]


def stratified_indices(rng, session_slices: list[tuple[int, int]], L: int, b: int) -> np.ndarray:
    """Global trial indices (b, n_total); each session resampled only within its own slice."""
    parts = []
    for start, stop in session_slices:
        parts.append(start + circular_block_positions(rng, stop - start, L, b))
    return np.concatenate(parts, axis=1)


def counts_from_indices(idx: np.ndarray, n_trials: int) -> np.ndarray:
    b = idx.shape[0]
    flat = (np.arange(b)[:, None] * n_trials + idx).ravel()
    return np.bincount(flat, minlength=b * n_trials).reshape(b, n_trials).astype(np.float64)


# ---------------------------------------------------------------------------- estimators
def pairwise_diffs(region_values: np.ndarray) -> np.ndarray:
    """(..., 4) -> (..., 6), column k = region a - region b for COMPARISONS[k]."""
    return np.stack([region_values[..., a] - region_values[..., b] for a, b in COMPARISONS], axis=-1)


def pooled_means(C, S, N, w):
    """Pooled mean sum(S)/sum(N) per region; trials weighted by counts C and inclusion w."""
    Sw = S * w[:, None]
    Nw = N * w[:, None]
    num = C @ Sw
    den = C @ Nw
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = num / den
    mu[den == 0] = np.nan
    return mu


def session_equal_means(C, V, w, session_masks):
    """Trial-equal mean within each session (missing V skipped), then sessions equal-weighted.

    V: (T, K) with NaN for missing. Returns (b, K); NaN if a session has no usable trial.
    """
    M = np.isfinite(V) & (w[:, None] > 0)
    V0 = np.where(M, V, 0.0)
    out = []
    for sm in session_masks:
        Cs = C[:, sm]
        num = Cs @ V0[sm]
        den = Cs @ M[sm].astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            val = num / den
        val[den == 0] = np.nan
        out.append(val)
    return np.mean(np.stack(out, axis=0), axis=0)


def adjusted_regression(C, D6, X, session_cols):
    """Weighted OLS per resample and comparison; returns mean of session intercepts (b, 6).

    D6: (T, 6) trial-level differences (NaN = missing). X: (T, k) design whose first
    ``session_cols`` columns are session indicators. C supplies frequency weights.
    """
    b = C.shape[0]
    out = np.full((b, D6.shape[1]), np.nan)
    rank_def = 0
    for k in range(D6.shape[1]):
        m = np.isfinite(D6[:, k])
        Xk = X[m]
        yk = D6[m, k]
        Ck = C[:, m]
        XtX = np.einsum("bt,ti,tj->bij", Ck, Xk, Xk, optimize=True)
        Xty = np.einsum("bt,ti,t->bi", Ck, Xk, yk, optimize=True)
        cond = np.linalg.cond(XtX)
        ok = cond < 1e10
        rank_def += int((~ok).sum())
        if ok.any():
            beta = np.linalg.solve(XtX[ok], Xty[ok][..., None])[..., 0]
            out[ok, k] = beta[:, :session_cols].mean(axis=1)
    return out, rank_def


# ---------------------------------------------------------------------------- inference
def summarize(delta_hat: np.ndarray, boot: np.ndarray, n_tests: int = N_COMP):
    """Centered absolute-error bootstrap test and symmetric intervals for each column."""
    rows = []
    for k in range(boot.shape[1]):
        d = float(delta_hat[k])
        e = boot[:, k] - d
        e = e[np.isfinite(e)]
        B = int(e.size)
        if B == 0 or not np.isfinite(d):
            rows.append(dict(difference_ms=d, bootstrap_B_used=B, n_nonfinite_resamples=int(boot.shape[0] - B),
                             raw_p=np.nan, adjusted_p=np.nan, p_mc_se=np.nan, n_exceed=np.nan,
                             ci95_low=np.nan, ci95_high=np.nan, ci_bonf_low=np.nan, ci_bonf_high=np.nan,
                             boot_se=np.nan, boot_bias=np.nan))
            continue
        ae = np.abs(e)
        n_exceed = int(np.sum(ae >= abs(d)))
        p = (1 + n_exceed) / (B + 1)
        q95 = float(np.quantile(ae, 0.95))
        qb = float(np.quantile(ae, 1 - 0.05 / n_tests))
        rows.append(dict(
            difference_ms=d, bootstrap_B_used=B, n_nonfinite_resamples=int(boot.shape[0] - B),
            raw_p=p, adjusted_p=min(n_tests * p, 1.0), p_mc_se=float(np.sqrt(p * (1 - p) / B)),
            n_exceed=n_exceed, ci95_low=d - q95, ci95_high=d + q95,
            ci_bonf_low=d - qb, ci_bonf_high=d + qb,
            boot_se=float(np.std(boot[:, k][np.isfinite(boot[:, k])], ddof=1)),
            boot_bias=float(np.mean(e)),
        ))
    return rows


def near_threshold(adjusted_p: float, p_mc_se: float, n_tests: int = N_COMP,
                   thresholds=(0.05, 0.01, 0.001)) -> bool:
    if not np.isfinite(adjusted_p):
        return False
    band = 3 * n_tests * p_mc_se
    return any(abs(adjusted_p - t) < band for t in thresholds)


def stars(p_adj: float) -> str:
    if not np.isfinite(p_adj):
        return "n/a"
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    if p_adj < 0.05:
        return "*"
    return "ns"
