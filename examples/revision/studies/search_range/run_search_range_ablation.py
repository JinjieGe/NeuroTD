#!/usr/bin/env python3
"""Discrete-simulation search-range ablation: |t|<=floor(W/4) vs |t|<=floor(W/2).

Does not modify paper sources or existing audit outputs. Re-optimizes each
candidate set. Analysis A shares the original half-range window grid. Analysis B
rebuilds the valid-window grid with each range's own margin.

Rerun from the repository root:

    MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mpl-neurotd \\
    PYTHONPATH=src \\
    python \\
    reproduction_audit/discrete_search_range_ablation_20260926/run_search_range_ablation.py

Resume is automatic: a seed with status ok is skipped. Pass --force to recompute.
"""

from __future__ import annotations
from revision_runtime import resolve_path

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# Headless, per-process matplotlib config. Workers inherit the env.
_OUT_DIR = Path(__file__).resolve().parent
_SRC = Path(resolve_path("repo", "src"))
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_OUT_DIR / ".mplconfig"))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from neurotd.neuro_signal import SegmentedSignal, my_ricker
from neurotd.time_varying_shift import (
    CTWShift,
    DTWShift,
    FFTShift,
    SignalParameters,
    compute_shifts_sliding_window_gt,
    compute_windowed_mse_with_hop,
    evaluate_alignment_quality_sliding,
)
from neurotd.utils import SlidingWindowPair, rmse

FS = 5120.0
DURATION_S = 1.0
N_SEGMENTS = 10
SEED_START = 0
SEED_STOP = 29  # inclusive
N_EXPECTED_SEEDS = SEED_STOP - SEED_START + 1

# Active grid in fig2_simulation_discrete_shift.py. Prior reproduction of
# supplementary table S1 used this grid. It is not relabeled here.
CURRENT_WINDOWS = np.array([33, 65, 129, 257, 513, 1025], dtype=int)
CURRENT_NOISE = np.linspace(0.0, 0.2, 11)

# Commented legacy grid in the same script ("before 2026.01.16").
# Not treated as a confirmed submission setting.
HISTORICAL_WINDOWS = np.array([51, 119, 171, 257, 513, 1281], dtype=int)
HISTORICAL_NOISE = np.linspace(0.0, 0.2, 10)

GRIDS = {
    "current": {
        "windows": CURRENT_WINDOWS,
        "noise_stds": CURRENT_NOISE,
        "role": "active_code_grid_used_by_prior_S1_reproduction",
    },
    "historical_comment": {
        "windows": HISTORICAL_WINDOWS,
        "noise_stds": HISTORICAL_NOISE,
        "role": "commented_legacy_grid_not_confirmed_as_submission",
    },
}

PRIOR_CSV = Path(resolve_path("baseline_tables", "simulation_results_discrete.csv"))


def sha256_bytes(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return hashlib.sha256(a.view(np.uint8)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sample_to_ms(samples, fs: float = FS) -> np.ndarray:
    return np.asarray(samples, dtype=np.float64) * (1000.0 / fs)


def build_signal():
    paras = SignalParameters()
    sig = SegmentedSignal(
        duration=DURATION_S,
        sampling_rate=FS,
        n_segments=N_SEGMENTS,
        shape_func=my_ricker,
    )
    shifted = sig.generate_shifted(paras.shifts)
    seg_len = sig.n_samples_per_seg
    seg_centers = seg_len // 2 + seg_len * np.arange(sig.n_segments)
    return paras, sig, shifted, seg_centers


def draw_standard_noise(seed: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """Match generate_noisy called twice after default_rng(seed).

    rng.normal(scale=sigma) equals sigma * rng.normal(). Resetting the seed
    for every window, as fig2 does, makes the standard normal draws identical
    across window sizes and noise levels.
    """
    rng = np.random.default_rng(seed)
    zx = rng.normal(size=n_samples)
    zy = rng.normal(size=n_samples)
    return zx, zy


def assert_noise_equivalence(seed: int, n_samples: int, noise_std: float) -> None:
    rng = np.random.default_rng(seed)
    a = rng.normal(scale=noise_std, size=n_samples)
    b = rng.normal(scale=noise_std, size=n_samples)
    zx, zy = draw_standard_noise(seed, n_samples)
    if not np.allclose(a, noise_std * zx, rtol=0, atol=0) or not np.allclose(b, noise_std * zy, rtol=0, atol=0):
        # Numpy may scale inside or outside the draw. Fall back to the same two calls
        # used by SegmentedSignal.generate_noisy if the algebraic form is not bitwise.
        rng_ref = np.random.default_rng(seed)
        ref_x = rng_ref.normal(scale=noise_std, size=n_samples)
        ref_y = rng_ref.normal(scale=noise_std, size=n_samples)
        if not (np.array_equal(a, ref_x) and np.array_equal(b, ref_y)):
            raise RuntimeError("repeated default_rng(seed) draws are not stable")
        raise RuntimeError(
            "noise_std * standard_normal is not bitwise-equal to normal(scale=noise_std); "
            "refusing to share one draw across noise levels"
        )


def window_starts(n: int, window: int, hop: int, margin: int) -> np.ndarray:
    stop = n - window - margin + 1
    return np.arange(margin, stop, hop, dtype=np.int64)


def search_mse(x: np.ndarray, y: np.ndarray, window: int, starts: np.ndarray, radius: int):
    """Rectangle-window weighted MSE. Tie keeps the smallest tau (strict <).

    Reported delay is -tau, matching compute_windowed_mse_with_hop.
    """
    if len(starts) == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        empty_f = np.zeros(0, dtype=np.float64)
        return empty_i, empty_i, empty_f
    weights = np.ones(window, dtype=np.float64)
    weights /= np.sum(weights)
    taus = np.arange(-radius, radius + 1, dtype=np.int64)
    cols = np.arange(window, dtype=np.int64)
    x_idx = starts[:, None] + cols[None, :]
    x_seg = x[x_idx]
    mse = np.empty((len(starts), len(taus)), dtype=np.float64)
    for j, tau in enumerate(taus):
        y_seg = y[x_idx - int(tau)]
        diff = x_seg - y_seg
        mse[:, j] = np.sum(weights * (diff * diff), axis=1)
    best = np.argmin(mse, axis=1)
    min_mse = mse[np.arange(len(starts)), best]
    reported = -taus[best]
    centers = starts + (window - 1) / 2.0
    centers = np.array(centers, dtype=int)
    return centers, reported.astype(np.int64), min_mse


def reference_search(x, y, window, hop, radius, margin):
    """Literal loop copied from compute_windowed_mse_with_hop, with an explicit radius and margin."""
    n = len(x)
    weights = np.ones(window, dtype=float)
    weights /= np.sum(weights)
    starts = range(int(margin), n - window - int(margin) + 1, int(hop))
    centers = []
    shifts = []
    min_mses = []
    for k in starts:
        centers.append(k + (window - 1) / 2)
        x_seg = x[k : k + window]
        best_tau = 0
        min_mse = np.inf
        for tau in range(-int(radius), int(radius) + 1):
            y_start = k - tau
            y_end = y_start + window
            if 0 <= y_start and y_end <= n:
                y_seg = y[y_start:y_end]
                diff = x_seg - y_seg
                weighted_mse = np.sum(weights * (diff * diff))
                if weighted_mse < min_mse:
                    min_mse = weighted_mse
                    best_tau = tau
            else:
                raise RuntimeError(f"invalid indices k={k} tau={tau} W={window}")
        shifts.append(-best_tau)
        min_mses.append(min_mse)
    return np.array(centers, dtype=int), np.asarray(shifts, dtype=np.int64), np.asarray(min_mses, dtype=np.float64)


def mixed_boundary_mask(starts: np.ndarray, window: int, seg_len: int, n_samples: int) -> np.ndarray:
    edges = np.arange(seg_len, n_samples, seg_len)
    if len(starts) == 0 or len(edges) == 0:
        return np.zeros(len(starts), dtype=bool)
    return np.any((starts[:, None] < edges[None, :]) & (edges[None, :] < (starts[:, None] + window)), axis=1)


def subset_metrics(err: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.sqrt(np.mean(err[mask] ** 2)))


def evaluate_condition(x, y, window, centers, shifts, gt, min_mse, starts, seg_len, shifts_true, fs):
    quality_e, quality_per = evaluate_alignment_quality_sliding(x, y, shifts, centers, window)
    if np.ndim(quality_per) == 0 or len(np.atleast_1d(quality_per)) != len(shifts):
        raise RuntimeError(
            f"quality window count {np.size(quality_per)} != estimate count {len(shifts)} at W={window}"
        )
    nrmse = np.asarray(quality_per, dtype=np.float64) ** 2
    e_from_parts = float(np.sqrt(np.mean(nrmse)))
    if not np.isclose(e_from_parts, quality_e, rtol=1e-10, atol=1e-12):
        raise RuntimeError(f"E identity failed: {e_from_parts} vs {quality_e}")
    # Paper formula (6) is a different functional of per-window NRMSE. Not used for selection.
    paper_q = float(1.0 / (1.0 + np.sqrt(np.mean(nrmse**2))))
    err = shifts.astype(np.float64) - gt.astype(np.float64)
    radius_note = None
    mixed = mixed_boundary_mask(starts, window, seg_len, len(x))
    center_segment = (centers // seg_len).astype(np.int64)
    seg_centers = seg_len // 2 + seg_len * np.arange(len(shifts_true))
    gt_segment = np.argmin(np.abs(centers.astype(np.float64)[:, None] - seg_centers[None, :]), axis=1).astype(np.int64)
    if not np.array_equal(gt.astype(np.int64), shifts_true[gt_segment].astype(np.int64)):
        raise RuntimeError("center-mapped ground truth does not match nearest segment center")
    sample_delay = np.repeat(np.asarray(shifts_true, dtype=np.int64), seg_len)
    cols = np.arange(window, dtype=np.int64)
    sample_idx = starts[:, None] + cols[None, :]
    frac_samples = np.mean(sample_delay[sample_idx] != gt[:, None], axis=1)
    return {
        "quality_e": float(quality_e),
        "paper_q": paper_q,
        "rmse": float(rmse(gt, shifts)),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
        "n_windows": int(len(shifts)),
        "mean_local_min_mse": float(np.mean(min_mse)) if len(min_mse) else float("nan"),
        "n_mixed": int(np.sum(mixed)),
        "rmse_interior": subset_metrics(err, ~mixed),
        "rmse_mixed": subset_metrics(err, mixed),
        "mean_frac_samples_not_center_gt": float(np.mean(frac_samples)) if len(frac_samples) else float("nan"),
        "n_gt_segment_ne_center_segment": int(np.sum(gt_segment != center_segment)),
        "mixed": mixed,
        "gt_segment": gt_segment,
        "center_segment": center_segment,
        "radius_note": radius_note,
    }


def baseline_once(x, y, window, hop, seg_centers, shifts_true):
    t0 = time.perf_counter()
    pair = SlidingWindowPair(x, y, window, hop)
    shifts_dtw = np.asarray(DTWShift(pair).shifts, dtype=np.float64)
    shifts_ctw = np.asarray(CTWShift(pair).shifts, dtype=np.float64)
    shifts_fft = np.asarray(FFTShift(pair, is_filter=False).shifts, dtype=np.float64)
    gt = np.asarray(
        compute_shifts_sliding_window_gt(pair.center_indices, seg_centers, shifts_true),
        dtype=np.float64,
    )
    elapsed = time.perf_counter() - t0
    return {
        "rmse_dtw": float(rmse(gt, shifts_dtw)),
        "rmse_ctw": float(rmse(gt, shifts_ctw)),
        "rmse_fft": float(rmse(gt, shifts_fft)),
        "n_windows_baseline": int(pair.num_windows),
        "baseline_runtime_s": elapsed,
        "baseline_center0": float(pair.center_indices[0]),
        "baseline_center_last": float(pair.center_indices[-1]),
    }


def delay_metrics(shifts, gt, radius: int):
    gt = np.asarray(gt, dtype=np.float64)
    shifts = np.asarray(shifts, dtype=np.float64)
    out = np.abs(gt) > radius
    edge = np.abs(shifts) == radius
    excess = np.maximum(np.abs(gt) - radius, 0.0)
    lower = float(np.sqrt(np.mean(excess**2))) if len(gt) else float("nan")
    return {
        "frac_gt_out_of_range": float(np.mean(out)) if len(gt) else float("nan"),
        "frac_estimate_on_edge": float(np.mean(edge)) if len(shifts) else float("nan"),
        "frac_estimate_on_pos_edge": float(np.mean(shifts == radius)) if len(shifts) else float("nan"),
        "frac_estimate_on_neg_edge": float(np.mean(shifts == -radius)) if len(shifts) else float("nan"),
        "rmse_lower_bound": lower,
    }


def process_seed(task: dict) -> dict:
    grid_name = task["grid_name"]
    seed = int(task["seed"])
    windows = np.asarray(task["windows"], dtype=int)
    noises = np.asarray(task["noise_stds"], dtype=np.float64)
    out_dir = Path(task["out_dir"])
    verify = bool(task["verify"])
    fs = float(task["fs"])
    ckpt = out_dir / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    status_path = ckpt / f"{grid_name}_seed{seed:02d}.json"
    summary_path = ckpt / f"{grid_name}_seed{seed:02d}_summary.csv"
    windows_path = ckpt / f"{grid_name}_seed{seed:02d}_windows.npz"
    if status_path.exists() and not task.get("force"):
        prev = json.loads(status_path.read_text())
        if prev.get("status") == "ok" and summary_path.exists() and windows_path.exists():
            prev["skipped"] = True
            return prev
    t_seed = time.perf_counter()
    try:
        paras, sig, shifted, seg_centers = build_signal()
        if sig.n_samples != int(round(DURATION_S * fs)):
            raise RuntimeError(f"unexpected length {sig.n_samples}")
        if int(sig.sampling_rate) != int(fs):
            raise RuntimeError("sampling rate mismatch")
        zx, zy = draw_standard_noise(seed, sig.n_samples)
        if seed == 0 and float(noises[0]) == 0.0:
            assert_noise_equivalence(seed, sig.n_samples, float(noises[min(1, len(noises) - 1)]))
        rows = []
        win_pack = {k: [] for k in (
            "noise_index", "window_index", "analysis", "range_code", "center", "k_start",
            "shift", "gt", "local_min_mse", "mixed", "gt_segment", "center_segment",
        )}
        verification = []
        seg_len = int(sig.n_samples_per_seg)
        shifts_true = np.asarray(paras.shifts, dtype=np.int64)
        for noise_index, noise_std in enumerate(noises):
            x = np.asarray(sig.data + float(noise_std) * zx, dtype=np.float64)
            y = np.asarray(shifted.data + float(noise_std) * zy, dtype=np.float64)
            if float(noise_std) == 0.0 and not np.array_equal(x, np.asarray(sig.data, dtype=np.float64)):
                raise RuntimeError("zero-noise reference was not identical to the clean signal")
            x_hash = sha256_bytes(x)
            y_hash = sha256_bytes(y)
            for window_index, window in enumerate(windows):
                window = int(window)
                hop = window // 2
                radius_q = window // 4
                radius_h = window // 2
                starts_h = window_starts(len(x), window, hop, radius_h)
                starts_q = window_starts(len(x), window, hop, radius_q)
                base = baseline_once(x, y, window, hop, seg_centers, shifts_true)
                searches = {}
                for key, starts, radius in (
                    ("half", starts_h, radius_h),
                    ("quarter_on_half_grid", starts_h, radius_q),
                    ("quarter_on_quarter_grid", starts_q, radius_q),
                ):
                    t0 = time.perf_counter()
                    centers, est, min_mse = search_mse(x, y, window, starts, radius)
                    searches[key] = {
                        "centers": centers,
                        "starts": starts,
                        "shifts": est,
                        "min_mse": min_mse,
                        "runtime_s": time.perf_counter() - t0,
                        "radius": radius,
                    }
                # Analysis A half and analysis B half are the same call.
                specs = (
                    ("A", "quarter", searches["quarter_on_half_grid"], radius_q, radius_h),
                    ("A", "half", searches["half"], radius_h, radius_h),
                    ("B", "quarter", searches["quarter_on_quarter_grid"], radius_q, radius_q),
                    ("B", "half", searches["half"], radius_h, radius_h),
                )
                packed = {}
                for analysis, range_name, pack, radius, margin in specs:
                    gt = np.asarray(
                        compute_shifts_sliding_window_gt(pack["centers"], seg_centers, shifts_true),
                        dtype=np.int64,
                    )
                    ev = evaluate_condition(
                        x, y, window, pack["centers"], pack["shifts"], gt, pack["min_mse"],
                        pack["starts"], seg_len, shifts_true, fs,
                    )
                    dm = delay_metrics(pack["shifts"], gt, radius)
                    packed[(analysis, range_name)] = (pack, gt, ev, dm, radius, margin)
                # Shared-center pairing by center value, never by row position.
                for analysis in ("A", "B"):
                    pq, gtq, evq, dmq, rq, mq = packed[(analysis, "quarter")]
                    ph, gth, evh, dmh, rh, mh = packed[(analysis, "half")]
                    map_h = {int(c): i for i, c in enumerate(ph["centers"])}
                    shared_q = []
                    shared_h = []
                    for i, c in enumerate(pq["centers"]):
                        j = map_h.get(int(c))
                        if j is not None:
                            shared_q.append(i)
                            shared_h.append(j)
                    shared_q = np.asarray(shared_q, dtype=int)
                    shared_h = np.asarray(shared_h, dtype=int)
                    if len(shared_q):
                        disagree = pq["shifts"][shared_q] != ph["shifts"][shared_h]
                        frac_disagree = float(np.mean(disagree))
                        outside = np.abs(ph["shifts"][shared_h]) > rq
                        frac_half_outside = float(np.mean(outside))
                        mse_gap = ph["min_mse"][shared_h] - pq["min_mse"][shared_q]
                        n_mse_violation = int(np.sum(mse_gap > 1e-8))
                        max_mse_gap = float(np.max(mse_gap))
                    else:
                        frac_disagree = float("nan")
                        frac_half_outside = float("nan")
                        n_mse_violation = 0
                        max_mse_gap = float("nan")
                    if analysis == "A":
                        if not np.array_equal(pq["centers"], ph["centers"]):
                            raise RuntimeError("analysis A window centers diverged")
                        if n_mse_violation:
                            raise RuntimeError(
                                f"half local MSE exceeded quarter at seed={seed} noise={noise_std} W={window}"
                            )
                    pair_info = {
                        "n_shared_centers": int(len(shared_q)),
                        "frac_estimates_disagree": frac_disagree,
                        "frac_half_outside_quarter": frac_half_outside,
                        "n_mse_violation": n_mse_violation,
                        "max_mse_half_minus_quarter": max_mse_gap,
                    }
                    for range_name in ("quarter", "half"):
                        pack, gt, ev, dm, radius, margin = packed[(analysis, range_name)]
                        row = {
                            "grid": grid_name,
                            "seed": seed,
                            "noise_index": noise_index,
                            "noise_std": float(noise_std),
                            "window_index": window_index,
                            "window_size": window,
                            "hop_size": hop,
                            "analysis": analysis,
                            "search_range": range_name,
                            "radius": int(radius),
                            "margin": int(margin),
                            "fs_hz": fs,
                            "ms_per_sample": 1000.0 / fs,
                            "n_windows": ev["n_windows"],
                            "center_first": int(pack["centers"][0]) if len(pack["centers"]) else -1,
                            "center_last": int(pack["centers"][-1]) if len(pack["centers"]) else -1,
                            "k_first": int(pack["starts"][0]) if len(pack["starts"]) else -1,
                            "k_last": int(pack["starts"][-1]) if len(pack["starts"]) else -1,
                            "rmse_samples": ev["rmse"],
                            "mae_samples": ev["mae"],
                            "bias_samples": ev["bias"],
                            "rmse_ms": float(sample_to_ms(ev["rmse"], fs)),
                            "mae_ms": float(sample_to_ms(ev["mae"], fs)),
                            "bias_ms": float(sample_to_ms(ev["bias"], fs)),
                            "quality_e": ev["quality_e"],
                            "paper_q_not_used": ev["paper_q"],
                            "mean_local_min_mse": ev["mean_local_min_mse"],
                            "neurotd_runtime_s": pack["runtime_s"],
                            "n_mixed": ev["n_mixed"],
                            "frac_mixed": ev["n_mixed"] / ev["n_windows"] if ev["n_windows"] else float("nan"),
                            "rmse_interior_samples": ev["rmse_interior"],
                            "rmse_mixed_samples": ev["rmse_mixed"],
                            "rmse_interior_ms": float(sample_to_ms(ev["rmse_interior"], fs)),
                            "rmse_mixed_ms": float(sample_to_ms(ev["rmse_mixed"], fs)),
                            "mean_frac_samples_not_center_gt": ev["mean_frac_samples_not_center_gt"],
                            "n_gt_segment_ne_center_segment": ev["n_gt_segment_ne_center_segment"],
                            "x_sha256": x_hash,
                            "y_sha256": y_hash,
                            **dm,
                            **pair_info,
                            **base,
                        }
                        row["rmse_dtw_ms"] = float(sample_to_ms(base["rmse_dtw"], fs))
                        row["rmse_ctw_ms"] = float(sample_to_ms(base["rmse_ctw"], fs))
                        row["rmse_fft_ms"] = float(sample_to_ms(base["rmse_fft"], fs))
                        row["rmse_lower_bound_ms"] = float(sample_to_ms(dm["rmse_lower_bound"], fs))
                        rows.append(row)
                        code_a = 0 if analysis == "A" else 1
                        code_r = 0 if range_name == "quarter" else 1
                        n = len(pack["centers"])
                        win_pack["noise_index"].append(np.full(n, noise_index, dtype=np.int16))
                        win_pack["window_index"].append(np.full(n, window_index, dtype=np.int16))
                        win_pack["analysis"].append(np.full(n, code_a, dtype=np.int8))
                        win_pack["range_code"].append(np.full(n, code_r, dtype=np.int8))
                        win_pack["center"].append(pack["centers"].astype(np.int32))
                        win_pack["k_start"].append(pack["starts"].astype(np.int32))
                        win_pack["shift"].append(pack["shifts"].astype(np.int16))
                        win_pack["gt"].append(gt.astype(np.int16))
                        win_pack["local_min_mse"].append(pack["min_mse"].astype(np.float64))
                        win_pack["mixed"].append(ev["mixed"].astype(np.int8))
                        win_pack["gt_segment"].append(ev["gt_segment"].astype(np.int8))
                        win_pack["center_segment"].append(ev["center_segment"].astype(np.int8))

                if verify:
                    lib_centers, lib_shifts = compute_windowed_mse_with_hop(
                        x, y, window, hop, window_fun="rectangle", continuous=False
                    )
                    half = searches["half"]
                    if not np.array_equal(half["centers"], np.asarray(lib_centers)) or not np.array_equal(
                        half["shifts"], np.asarray(lib_shifts)
                    ):
                        raise RuntimeError(
                            f"vectorized half != compute_windowed_mse_with_hop at noise={noise_std} W={window}"
                        )
                    for label, starts, radius, margin in (
                        ("quarter_A", starts_h, radius_q, radius_h),
                        ("quarter_B", starts_q, radius_q, radius_q),
                        ("half", starts_h, radius_h, radius_h),
                    ):
                        ref_c, ref_s, ref_m = reference_search(x, y, window, hop, radius, margin)
                        got = search_mse(x, y, window, starts, radius)
                        if not (
                            np.array_equal(got[0], ref_c)
                            and np.array_equal(got[1], ref_s)
                            and np.allclose(got[2], ref_m, rtol=0, atol=0)
                        ):
                            # Integer shifts are the estimate. MSE must match closely if a sum-order difference appears.
                            if not (np.array_equal(got[0], ref_c) and np.array_equal(got[1], ref_s)):
                                raise RuntimeError(f"vectorized != reference loop for {label} W={window}")
                            if not np.allclose(got[2], ref_m, rtol=1e-12, atol=1e-12):
                                raise RuntimeError(f"MSE mismatch for {label} W={window}")
                    verification.append({"noise_index": noise_index, "window": window, "half_matches_library": True})

        summary = pd.DataFrame(rows)
        summary.to_csv(summary_path, index=False)
        arrays = {k: np.concatenate(v) for k, v in win_pack.items()}
        np.savez_compressed(windows_path, **arrays)
        elapsed = time.perf_counter() - t_seed
        status = {
            "status": "ok",
            "grid": grid_name,
            "seed": seed,
            "n_rows": int(len(summary)),
            "n_window_rows": int(len(arrays["center"])),
            "runtime_s": elapsed,
            "verify": verify,
            "n_verified_conditions": len(verification),
            "zx_sha256": sha256_bytes(zx),
            "zy_sha256": sha256_bytes(zy),
            "clean_x_sha256": sha256_bytes(np.asarray(sig.data)),
            "shifts": shifts_true.tolist(),
            "n_samples": int(sig.n_samples),
            "seg_len": seg_len,
            "fs_hz": fs,
            "signalparameters_sampling_rate": float(paras.sampling_rate),
            "skipped": False,
        }
        status_path.write_text(json.dumps(status, indent=2))
        print(
            f"done grid={grid_name} seed={seed} rows={len(summary)} windows={len(arrays['center'])} "
            f"seconds={elapsed:.1f} verify={verify}",
            flush=True,
        )
        return status
    except Exception as exc:
        err = {"status": "error", "grid": grid_name, "seed": seed, "error": str(exc), "traceback": traceback.format_exc()}
        status_path.write_text(json.dumps(err, indent=2))
        print(f"FAILED grid={grid_name} seed={seed}: {exc}", flush=True)
        return err


def source_environment() -> dict:
    import dtw
    import scipy
    import sklearn
    import tslearn
    import skimage

    # Hash by path. Do not import fig2_simulation_discrete_shift: that file runs the
    # full simulation, writes figures, and blocks on plt.show at import time.
    root = Path(resolve_path("repo", "src/neurotd"))
    files = {
        "fig2_simulation_discrete_shift.py": Path(resolve_path("repo", "examples/simulations/fig2_simulation_discrete_shift.py")),
        "time_varying_shift.py": root / "time_varying_shift.py",
        "neuro_signal.py": root / "neuro_signal.py",
        "utils.py": root / "utils.py",
        "simulation_plotting.py": root / "simulation_plotting.py",
    }
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "dtw": getattr(dtw, "__version__", "unknown"),
        "tslearn": tslearn.__version__,
        "skimage": skimage.__version__,
        "neurotd_time_varying_shift": str(files["time_varying_shift.py"]),
        "source_sha256": {name: sha256_file(path) for name, path in files.items()},
        "git_repo": (Path(resolve_path("repo")) / ".git").exists(),
        "git_note": (
            "This is a portable integration of the historical ZIP-based experiment. "
            "Use current source hashes and the launcher manifest for this run; "
            "the historical snapshot was identified as 1a6c4f23818bcfe3152f947c54c553a92edeca0c."
        ),
        "fs_hz_explicit": FS,
        "duration_s": DURATION_S,
        "n_segments": N_SEGMENTS,
        "waveform": "my_ricker (Mexican-hat / Ricker), a = N//16",
        "window": "rectangle, continuous=False",
        "hop": "W//2",
        "tie_break": "strict < on weighted MSE, tau scanned from -L to +L, so the most negative tau wins ties; reported delay is -tau",
        "quality_E": "sqrt(mean(per-window NRMSE)) from evaluate_alignment_quality_sliding; selection is argmin",
        "paper_q_not_used": "1/(1+sqrt(mean(NRMSE^2))); stored only as a side column",
    }


def compare_seed0_to_prior_csv(out_dir: Path) -> dict:
    summary = pd.read_csv(out_dir / "checkpoints" / "current_seed00_summary.csv")
    half = summary[(summary.analysis == "A") & (summary.search_range == "half") & (summary.seed == 0)].copy()
    prior = pd.read_csv(
        PRIOR_CSV,
        usecols=["noise_std", "window_size", "rmse_neurotd", "quality_rmse_neurotd", "rmse_dtw", "rmse_ctw", "rmse_fft"],
    )
    half = half.copy()
    prior = prior.copy()
    half["noise_key"] = np.round(half["noise_std"].to_numpy(dtype=np.float64), 10)
    prior["noise_key"] = np.round(prior["noise_std"].to_numpy(dtype=np.float64), 10)
    merged = half.merge(prior, on=["noise_key", "window_size"], how="outer", indicator=True, suffixes=("_run", "_prior"))
    if not (merged["_merge"] == "both").all():
        raise RuntimeError("seed-0 current grid did not align with the prior discrete CSV keys")
    cols = {
        "rmse_samples": "rmse_neurotd",
        "quality_e": "quality_rmse_neurotd",
        "rmse_dtw_run": "rmse_dtw_prior",
        "rmse_ctw_run": "rmse_ctw_prior",
        "rmse_fft_run": "rmse_fft_prior",
    }
    report = {"n": int(len(merged)), "max_abs": {}}
    tolerances = {
        "rmse_samples": 1e-8,
        "quality_e": 1e-8,
        "rmse_dtw_run": 1e-5,
        "rmse_ctw_run": 1e-5,
        "rmse_fft_run": 1e-5,
    }
    for left, right in cols.items():
        diff = np.abs(merged[left].to_numpy(dtype=float) - merged[right].to_numpy(dtype=float))
        report["max_abs"][f"{left}_vs_{right}"] = float(np.max(diff))
        if np.max(diff) > tolerances[left]:
            raise RuntimeError(f"prior CSV mismatch on {left}: max abs {np.max(diff)}")
    report["status"] = "match"
    return report


def write_parameter_audit(out_dir: Path) -> dict:
    paras, sig, shifted, seg_centers = build_signal()
    record = {
        "signalparameters_sample_per_seg_default": int(paras.sample_per_seg),
        "signalparameters_sampling_rate_default": float(paras.sampling_rate),
        "explicit_fs_hz": FS,
        "default_sampling_rate_equals_explicit_fs": bool(paras.sampling_rate == FS),
        "ms_per_sample": 1000.0 / FS,
        "n_samples": int(sig.n_samples),
        "n_samples_per_seg": int(sig.n_samples_per_seg),
        "shifts": [int(v) for v in paras.shifts],
        "shift_min": int(np.min(paras.shifts)),
        "shift_max": int(np.max(paras.shifts)),
        "ricker_width_a": int(sig.n_samples_per_seg // 16),
        "segment_centers": [int(v) for v in seg_centers],
        "grids": {},
        "other_discrete_conditions": (
            "fig2_simulation_discrete_shift.py is the paper discrete-shift script. "
            "fig3_simulation_continuous_shift.py is the continuous-delay experiment and is excluded. "
            "iEEG and Patch-seq scripts are excluded. The 40-segment diagnostic used in a later "
            "window-selection audit is not part of the paper discrete simulation and is not included. "
            "No third discrete grid is set in the active or commented fig2 code."
        ),
    }
    max_true = int(np.max(paras.shifts))
    for name, spec in GRIDS.items():
        rows = []
        for window in spec["windows"]:
            window = int(window)
            hop = window // 2
            rq, rh = window // 4, window // 2
            starts_h = window_starts(sig.n_samples, window, hop, rh)
            starts_q = window_starts(sig.n_samples, window, hop, rq)
            centers_h = np.array(starts_h + (window - 1) / 2.0, dtype=int)
            centers_q = np.array(starts_q + (window - 1) / 2.0, dtype=int)
            shared = len(set(centers_h.tolist()) & set(centers_q.tolist()))
            rows.append(
                {
                    "window": window,
                    "hop": hop,
                    "radius_quarter": rq,
                    "radius_half": rh,
                    "max_true_delay": max_true,
                    "quarter_cannot_reach_max_true": bool(max_true > rq),
                    "half_cannot_reach_max_true": bool(max_true > rh),
                    "n_windows_analysis_A": int(len(centers_h)),
                    "n_windows_analysis_B_quarter": int(len(centers_q)),
                    "n_shared_A_and_B_quarter_centers": int(shared),
                    "A_center_first": int(centers_h[0]),
                    "A_center_last": int(centers_h[-1]),
                    "B_quarter_center_first": int(centers_q[0]),
                    "B_quarter_center_last": int(centers_q[-1]),
                    "window_ms": float(window * 1000.0 / FS),
                }
            )
        record["grids"][name] = {
            "role": spec["role"],
            "windows": [int(v) for v in spec["windows"]],
            "noise_stds": [float(v) for v in spec["noise_stds"]],
            "per_window": rows,
        }
    (out_dir / "parameter_audit.json").write_text(json.dumps(record, indent=2))
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    out_dir = _OUT_DIR
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / ".mplconfig").mkdir(parents=True, exist_ok=True)
    if args.analyze_only:
        from analyze_search_range_ablation import analyze

        analyze(out_dir)
        return
    env = source_environment()
    (out_dir / "environment.json").write_text(json.dumps(env, indent=2))
    params = write_parameter_audit(out_dir)
    print("parameter audit written", flush=True)
    print(json.dumps({k: params[k] for k in ("shifts", "n_samples", "explicit_fs_hz", "default_sampling_rate_equals_explicit_fs")}, indent=2))

    tasks = []
    for grid_name, spec in GRIDS.items():
        for seed in range(SEED_START, SEED_STOP + 1):
            tasks.append(
                {
                    "grid_name": grid_name,
                    "seed": seed,
                    "windows": [int(v) for v in spec["windows"]],
                    "noise_stds": [float(v) for v in spec["noise_stds"]],
                    "out_dir": str(out_dir),
                    "verify": seed == 0,
                    "force": args.force,
                    "fs": FS,
                }
            )
    # Seed 0 of both grids is the consistency gate and runs first, in-process.
    gate_tasks = [t for t in tasks if t["seed"] == 0]
    later = [t for t in tasks if t["seed"] != 0]
    gate_results = []
    for task in gate_tasks:
        result = process_seed(task)
        gate_results.append(result)
        if result.get("status") != "ok":
            (out_dir / "verification_seed0.json").write_text(json.dumps({"gate": gate_results}, indent=2))
            raise SystemExit(f"seed 0 verification failed for {task['grid_name']}; remaining seeds were not started")
        if task["grid_name"] == "current":
            csv_report = compare_seed0_to_prior_csv(out_dir)
            print("seed 0 current half matches the prior CSV", csv_report["max_abs"], flush=True)
            (out_dir / "verification_seed0_current_csv.json").write_text(json.dumps(csv_report, indent=2))
    csv_report = compare_seed0_to_prior_csv(out_dir)
    (out_dir / "verification_seed0.json").write_text(
        json.dumps({"gate": gate_results, "prior_csv": csv_report}, indent=2)
    )
    print("seed 0 matches the prior half-range CSV", csv_report["max_abs"], flush=True)

    failed = []
    if later:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_seed, task): task for task in later}
            for fut in as_completed(futures):
                task = futures[fut]
                try:
                    result = fut.result()
                except Exception as exc:
                    result = {"status": "error", "grid": task["grid_name"], "seed": task["seed"], "error": str(exc)}
                if result.get("status") != "ok":
                    failed.append(result)
        # One retry for failures, still using the original estimator.
        if failed:
            print(f"retrying {len(failed)} failed seeds", flush=True)
            retry_tasks = []
            for item in failed:
                retry_tasks.extend(
                    t for t in later if t["grid_name"] == item.get("grid") and t["seed"] == item.get("seed")
                )
            failed = []
            for task in retry_tasks:
                task = dict(task)
                task["force"] = True
                result = process_seed(task)
                if result.get("status") != "ok":
                    failed.append(result)
    expected = []
    missing = []
    for grid_name in GRIDS:
        for seed in range(SEED_START, SEED_STOP + 1):
            status_path = out_dir / "checkpoints" / f"{grid_name}_seed{seed:02d}.json"
            expected.append(f"{grid_name}:{seed}")
            if not status_path.exists():
                missing.append(f"{grid_name}:{seed}:no_status")
                continue
            status = json.loads(status_path.read_text())
            if status.get("status") != "ok":
                missing.append(f"{grid_name}:{seed}:{status.get('status')}")
    completion = {
        "expected": len(expected),
        "missing_or_failed": missing,
        "failed_after_retry": failed,
        "complete": len(missing) == 0,
    }
    (out_dir / "completion_status.json").write_text(json.dumps(completion, indent=2))
    if missing:
        print("INCOMPLETE", missing, flush=True)
        raise SystemExit(2)
    from analyze_search_range_ablation import analyze

    analyze(out_dir)


if __name__ == "__main__":
    main()
