#!/usr/bin/env python3
"""Aggregate the discrete search-range ablation and write tables, figures, and the analysis report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FS = 5120.0
MS = 1000.0 / FS
SEG_LEN = 512
N_SEGMENTS = 10
SHIFTS = np.arange(8, 18, dtype=int)
SEG_CENTERS = (SEG_LEN // 2 + SEG_LEN * np.arange(N_SEGMENTS)).astype(int)
ANALYSIS_NAME = {0: "A", 1: "B"}
RANGE_NAME = {0: "quarter", 1: "half"}


def mean_ci(values: np.ndarray, alpha: float = 0.05):
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return dict(n=0, mean=np.nan, low=np.nan, high=np.nan, std=np.nan)
    mean = float(np.mean(x))
    if n == 1:
        return dict(n=1, mean=mean, low=np.nan, high=np.nan, std=np.nan)
    std = float(np.std(x, ddof=1))
    se = std / np.sqrt(n)
    tcrit = float(stats.t.ppf(1 - alpha / 2, n - 1))
    return dict(n=n, mean=mean, low=mean - tcrit * se, high=mean + tcrit * se, std=std)


def load_summary(out_dir: Path) -> pd.DataFrame:
    files = sorted((out_dir / "checkpoints").glob("*_summary.csv"))
    if not files:
        raise FileNotFoundError("no summary checkpoints")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df["analysis"] = df["analysis"].astype(str)
    df["search_range"] = df["search_range"].astype(str)
    return df


def load_windows(out_dir: Path, grid: str, seed: int) -> dict:
    path = out_dir / "checkpoints" / f"{grid}_seed{seed:02d}_windows.npz"
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def check_integrity(df: pd.DataFrame, out_dir: Path) -> dict:
    issues = []
    for (grid, seed, noise_index), g in df.groupby(["grid", "seed", "noise_index"], sort=False):
        if g["x_sha256"].nunique() != 1 or g["y_sha256"].nunique() != 1:
            issues.append(f"signal hash split {grid} seed={seed} noise_index={noise_index}")
    hashes = {}
    for path in sorted((out_dir / "checkpoints").glob("*_seed*.json")):
        status = json.loads(path.read_text())
        if status.get("status") != "ok":
            issues.append(f"status not ok: {path.name}")
            continue
        hashes.setdefault(int(status["seed"]), {})[status["grid"]] = status.get("zx_sha256")
    for seed, grids in hashes.items():
        vals = [v for v in grids.values() if v]
        if len(vals) >= 2 and len(set(vals)) != 1:
            issues.append(f"zx hash differs across grids at seed={seed}")
    mse_bad = df[(df.analysis == "A") & (df.n_mse_violation > 0)]
    if len(mse_bad):
        issues.append(f"analysis A MSE inequality violations: {len(mse_bad)} rows")
    # Ground-truth reachability does not depend on noise or seed.
    gt_var = (
        df.groupby(["grid", "analysis", "search_range", "window_size"])["frac_gt_out_of_range"].nunique()
    )
    if (gt_var > 1).any():
        issues.append("frac_gt_out_of_range varied across seeds or noise levels")
    return {"n_issues": len(issues), "issues": issues, "n_seeds_with_hash": len(hashes)}


def fixed_window_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["grid", "analysis", "noise_index", "noise_std", "window_size"]
    grouped = df.groupby(keys, sort=True)
    for key, g in grouped:
        q = g[g.search_range == "quarter"].sort_values("seed")
        h = g[g.search_range == "half"].sort_values("seed")
        if len(q) == 0 or len(h) == 0 or not np.array_equal(q.seed.to_numpy(), h.seed.to_numpy()):
            raise RuntimeError(f"unpaired seeds at {key}")
        diff = h.rmse_ms.to_numpy() - q.rmse_ms.to_numpy()
        ci = mean_ci(diff)
        rec = {
            "grid": key[0],
            "analysis": key[1],
            "noise_index": int(key[2]),
            "noise_std": float(q.noise_std.iloc[0]),
            "window_size": int(q.window_size.iloc[0]),
            "n_seeds": ci["n"],
            "mean_rmse_quarter_ms": float(q.rmse_ms.mean()),
            "mean_rmse_half_ms": float(h.rmse_ms.mean()),
            "mean_diff_half_minus_quarter_ms": ci["mean"],
            "ci95_low_ms": ci["low"],
            "ci95_high_ms": ci["high"],
            "ci_excludes_0": bool(np.isfinite(ci["low"]) and (ci["low"] > 0 or ci["high"] < 0)),
            "frac_seeds_half_lower_rmse": float(np.mean(diff < -1e-12)),
            "frac_seeds_equal_rmse": float(np.mean(np.abs(diff) <= 1e-12)),
            "mean_mae_quarter_ms": float(q.mae_ms.mean()),
            "mean_mae_half_ms": float(h.mae_ms.mean()),
            "mean_bias_quarter_ms": float(q.bias_ms.mean()),
            "mean_bias_half_ms": float(h.bias_ms.mean()),
            "mean_E_quarter": float(q.quality_e.mean()),
            "mean_E_half": float(h.quality_e.mean()),
            "mean_local_mse_quarter": float(q.mean_local_min_mse.mean()),
            "mean_local_mse_half": float(h.mean_local_min_mse.mean()),
            "frac_gt_out_quarter": float(q.frac_gt_out_of_range.iloc[0]),
            "frac_gt_out_half": float(h.frac_gt_out_of_range.iloc[0]),
            "lower_bound_quarter_ms": float(q.rmse_lower_bound_ms.iloc[0]),
            "lower_bound_half_ms": float(h.rmse_lower_bound_ms.iloc[0]),
            "mean_frac_edge_quarter": float(q.frac_estimate_on_edge.mean()),
            "mean_frac_edge_half": float(h.frac_estimate_on_edge.mean()),
            "mean_frac_neg_edge_half": float(h.frac_estimate_on_neg_edge.mean()),
            "mean_frac_half_outside_quarter": float(h.frac_half_outside_quarter.mean()),
            "mean_frac_disagree": float(q.frac_estimates_disagree.mean()),
            "mean_n_windows_quarter": float(q.n_windows.mean()),
            "mean_n_windows_half": float(h.n_windows.mean()),
            "mean_n_shared": float(q.n_shared_centers.mean()),
            "mean_frac_mixed_quarter": float(q.frac_mixed.mean()),
            "mean_frac_mixed_half": float(h.frac_mixed.mean()),
            "mean_rmse_interior_quarter_ms": float(q.rmse_interior_ms.mean()),
            "mean_rmse_mixed_quarter_ms": float(q.rmse_mixed_ms.mean()),
            "mean_rmse_interior_half_ms": float(h.rmse_interior_ms.mean()),
            "mean_rmse_mixed_half_ms": float(h.rmse_mixed_ms.mean()),
            "mean_frac_samples_not_gt_quarter": float(q.mean_frac_samples_not_center_gt.mean()),
            "mean_frac_samples_not_gt_half": float(h.mean_frac_samples_not_center_gt.mean()),
            "max_mse_violation": int(q.n_mse_violation.max()),
            "mean_runtime_quarter_s": float(q.neurotd_runtime_s.mean()),
            "mean_runtime_half_s": float(h.neurotd_runtime_s.mean()),
            "mean_baseline_runtime_s": float(q.baseline_runtime_s.mean()),
        }
        rows.append(rec)
    return pd.DataFrame(rows)


def _pick(g: pd.DataFrame, col: str) -> pd.Series:
    ordered = g.sort_values(["window_size", "window_index"])
    return ordered.iloc[int(np.argmin(ordered[col].to_numpy()))]


def selection_tables(df: pd.DataFrame):
    choices = []
    cross_rows = []
    rank_rows = []
    for (grid, analysis, seed, noise_index), g0 in df.groupby(
        ["grid", "analysis", "seed", "noise_index"], sort=True
    ):
        by_range = {}
        for range_name, g in g0.groupby("search_range"):
            quality = _pick(g, "quality_e")
            oracle = _pick(g, "rmse_samples")
            by_range[range_name] = (g, quality, oracle)
            q_at = g[g.window_size == int(quality.window_size)].iloc[0]
            choices.append(
                {
                    "grid": grid,
                    "analysis": analysis,
                    "seed": int(seed),
                    "noise_index": int(noise_index),
                    "noise_std": float(quality.noise_std),
                    "search_range": range_name,
                    "selected_W": int(quality.window_size),
                    "selected_E": float(quality.quality_e),
                    "selected_rmse_ms": float(quality.rmse_ms),
                    "selected_rmse_samples": float(quality.rmse_samples),
                    "oracle_W": int(oracle.window_size),
                    "oracle_rmse_ms": float(oracle.rmse_ms),
                    "oracle_E": float(oracle.quality_e),
                    "paper_q_selected_W": int(_pick(g, "paper_q_not_used")["window_size"])
                    if False
                    else int(g.sort_values("window_size").iloc[int(np.argmax(g.sort_values("window_size")["paper_q_not_used"].to_numpy()))].window_size),
                    "quality_matches_oracle": bool(int(quality.window_size) == int(oracle.window_size)),
                    "rmse_dtw_oracle_ms": float(g.rmse_dtw_ms.min()),
                    "rmse_ctw_oracle_ms": float(g.rmse_ctw_ms.min()),
                    "rmse_fft_oracle_ms": float(g.rmse_fft_ms.min()),
                    "rmse_dtw_at_selected_ms": float(q_at.rmse_dtw_ms),
                    "rmse_ctw_at_selected_ms": float(q_at.rmse_ctw_ms),
                    "rmse_fft_at_selected_ms": float(q_at.rmse_fft_ms),
                }
            )
            paper_w = int(
                g.sort_values("window_size").iloc[int(np.argmax(g.sort_values("window_size")["paper_q_not_used"].to_numpy()))].window_size
            )
            choices[-1]["paper_q_selected_W"] = paper_w
            choices[-1]["paper_q_differs_from_E"] = bool(paper_w != int(quality.window_size))
            scores_oracle_style = {
                "NeuroTD_quality": float(quality.rmse_ms),
                "DTW_oracle": float(g.rmse_dtw_ms.min()),
                "CTW_oracle": float(g.rmse_ctw_ms.min()),
                "FFT_oracle": float(g.rmse_fft_ms.min()),
            }
            scores_same_w = {
                "NeuroTD": float(quality.rmse_ms),
                "DTW": float(q_at.rmse_dtw_ms),
                "CTW": float(q_at.rmse_ctw_ms),
                "FFT": float(q_at.rmse_fft_ms),
            }
            rank_rows.append(
                {
                    "grid": grid,
                    "analysis": analysis,
                    "seed": int(seed),
                    "noise_index": int(noise_index),
                    "noise_std": float(quality.noise_std),
                    "search_range": range_name,
                    "selected_W": int(quality.window_size),
                    **{f"paper_style_{k}": v for k, v in scores_oracle_style.items()},
                    "paper_style_winner": min(scores_oracle_style, key=scores_oracle_style.get),
                    **{f"same_window_{k}": v for k, v in scores_same_w.items()},
                    "same_window_winner": min(scores_same_w, key=scores_same_w.get),
                    "neurotd_beats_all_oracle_baselines": bool(
                        scores_oracle_style["NeuroTD_quality"]
                        < min(scores_oracle_style["DTW_oracle"], scores_oracle_style["CTW_oracle"], scores_oracle_style["FFT_oracle"])
                    ),
                    "neurotd_beats_all_same_window": bool(
                        scores_same_w["NeuroTD"] < min(scores_same_w["DTW"], scores_same_w["CTW"], scores_same_w["FFT"])
                    ),
                }
            )
        gq, qq, oq = by_range["quarter"]
        gh, qh, oh = by_range["half"]
        wq, wh = int(qq.window_size), int(qh.window_size)

        def at(frame, window, col="rmse_ms"):
            return float(frame.loc[frame.window_size == window, col].iloc[0])

        cross_rows.append(
            {
                "grid": grid,
                "analysis": analysis,
                "seed": int(seed),
                "noise_index": int(noise_index),
                "noise_std": float(qq.noise_std),
                "W_quarter": wq,
                "W_half": wh,
                "same_selected_W": bool(wq == wh),
                "E_quarter": float(qq.quality_e),
                "E_half": float(qh.quality_e),
                "rmse_quarter_at_Wq_ms": at(gq, wq),
                "rmse_half_at_Wq_ms": at(gh, wq),
                "rmse_quarter_at_Wh_ms": at(gq, wh),
                "rmse_half_at_Wh_ms": at(gh, wh),
                "oracle_W_quarter": int(oq.window_size),
                "oracle_W_half": int(oh.window_size),
                "oracle_rmse_quarter_ms": float(oq.rmse_ms),
                "oracle_rmse_half_ms": float(oh.rmse_ms),
                "quality_rmse_quarter_ms": float(qq.rmse_ms),
                "quality_rmse_half_ms": float(qh.rmse_ms),
            }
        )
    return pd.DataFrame(choices), pd.DataFrame(cross_rows), pd.DataFrame(rank_rows)


def selection_summary(choices: pd.DataFrame, cross: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (grid, analysis, noise_index), c in choices.groupby(["grid", "analysis", "noise_index"], sort=True):
        cr = cross[(cross.grid == grid) & (cross.analysis == analysis) & (cross.noise_index == noise_index)]
        for range_name, g in c.groupby("search_range"):
            counts = g.selected_W.value_counts().to_dict()
            oracle_counts = g.oracle_W.value_counts().to_dict()
            end = cr.quality_rmse_quarter_ms if range_name == "quarter" else cr.quality_rmse_half_ms
            # paired end-to-end diff is computed once below
            rows.append(
                {
                    "grid": grid,
                    "analysis": analysis,
                    "noise_index": int(noise_index),
                    "noise_std": float(g.noise_std.iloc[0]),
                    "search_range": range_name,
                    "modal_selected_W": int(g.selected_W.mode().iloc[0]),
                    "frac_modal_selected_W": float((g.selected_W == g.selected_W.mode().iloc[0]).mean()),
                    "selection_counts": json.dumps({str(k): int(v) for k, v in sorted(counts.items())}),
                    "modal_oracle_W": int(g.oracle_W.mode().iloc[0]),
                    "oracle_counts": json.dumps({str(k): int(v) for k, v in sorted(oracle_counts.items())}),
                    "mean_selected_rmse_ms": float(g.selected_rmse_ms.mean()),
                    "mean_oracle_rmse_ms": float(g.oracle_rmse_ms.mean()),
                    "frac_quality_equals_oracle_W": float(g.quality_matches_oracle.mean()),
                    "frac_paper_q_differs_from_E": float(g.paper_q_differs_from_E.mean()),
                }
            )
        d_quality = cr.quality_rmse_half_ms.to_numpy() - cr.quality_rmse_quarter_ms.to_numpy()
        d_oracle = cr.oracle_rmse_half_ms.to_numpy() - cr.oracle_rmse_quarter_ms.to_numpy()
        cq, co = mean_ci(d_quality), mean_ci(d_oracle)
        rows.append(
            {
                "grid": grid,
                "analysis": analysis,
                "noise_index": int(noise_index),
                "noise_std": float(cr.noise_std.iloc[0]),
                "search_range": "paired_diff",
                "frac_same_selected_W": float(cr.same_selected_W.mean()),
                "mean_diff_quality_rmse_ms": cq["mean"],
                "ci95_low_quality_ms": cq["low"],
                "ci95_high_quality_ms": cq["high"],
                "quality_ci_excludes_0": bool(np.isfinite(cq["low"]) and (cq["low"] > 0 or cq["high"] < 0)),
                "mean_diff_oracle_rmse_ms": co["mean"],
                "ci95_low_oracle_ms": co["low"],
                "ci95_high_oracle_ms": co["high"],
                "oracle_ci_excludes_0": bool(np.isfinite(co["low"]) and (co["low"] > 0 or co["high"] < 0)),
                "n_seeds_different_W": int((~cr.same_selected_W).sum()),
                "mean_rmse_q_at_Wq_when_differ_ms": float(cr.loc[~cr.same_selected_W, "rmse_quarter_at_Wq_ms"].mean())
                if (~cr.same_selected_W).any()
                else np.nan,
                "mean_rmse_h_at_Wq_when_differ_ms": float(cr.loc[~cr.same_selected_W, "rmse_half_at_Wq_ms"].mean())
                if (~cr.same_selected_W).any()
                else np.nan,
                "mean_rmse_q_at_Wh_when_differ_ms": float(cr.loc[~cr.same_selected_W, "rmse_quarter_at_Wh_ms"].mean())
                if (~cr.same_selected_W).any()
                else np.nan,
                "mean_rmse_h_at_Wh_when_differ_ms": float(cr.loc[~cr.same_selected_W, "rmse_half_at_Wh_ms"].mean())
                if (~cr.same_selected_W).any()
                else np.nan,
            }
        )
    return pd.DataFrame(rows)


def centers_of(pack: dict, noise_index: int, window_index: int, analysis: int, range_code: int) -> np.ndarray:
    m = (
        (pack["noise_index"] == noise_index)
        & (pack["window_index"] == window_index)
        & (pack["analysis"] == analysis)
        & (pack["range_code"] == range_code)
    )
    return pack["center"][m]


def subset_rmse_ms(pack, noise_index, window_index, analysis, range_code, allowed_centers) -> float:
    m = (
        (pack["noise_index"] == noise_index)
        & (pack["window_index"] == window_index)
        & (pack["analysis"] == analysis)
        & (pack["range_code"] == range_code)
    )
    centers = pack["center"][m]
    keep = np.isin(centers, list(allowed_centers))
    if not np.any(keep):
        return float("nan")
    err = pack["shift"][m][keep].astype(np.float64) - pack["gt"][m][keep].astype(np.float64)
    return float(np.sqrt(np.mean(err**2)) * MS)


def map_segment_rmse_ms(pack, noise_index, window_index, analysis, range_code, query_times) -> float:
    m = (
        (pack["noise_index"] == noise_index)
        & (pack["window_index"] == window_index)
        & (pack["analysis"] == analysis)
        & (pack["range_code"] == range_code)
    )
    centers = pack["center"][m].astype(np.int64)
    shifts = pack["shift"][m].astype(np.float64)
    if len(centers) == 0:
        return float("nan")
    order = np.argsort(centers)
    centers = centers[order]
    shifts = shifts[order]
    lo, hi = int(centers[0]), int(centers[-1])
    est = []
    truth = []
    for t in query_times:
        t = int(t)
        if t < lo or t > hi:
            continue
        j = int(np.argmin(np.abs(centers - t)))
        est.append(shifts[j])
        truth.append(float(SHIFTS[t // SEG_LEN]))
    if not est:
        return float("nan")
    err = np.asarray(est) - np.asarray(truth)
    return float(np.sqrt(np.mean(err**2)) * MS)


def common_grid_tables(out_dir: Path, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coverage_rows = []
    for grid, g in df.groupby("grid"):
        seeds = sorted(g.seed.unique())
        windows = sorted(g.window_size.unique())
        noises = sorted(g.noise_index.unique())
        w_index = {int(w): int(i) for i, w in enumerate(windows)}
        # Centers are deterministic. Read them from seed 0.
        pack0 = load_windows(out_dir, grid, int(seeds[0]))
        center_sets_A = {}
        center_sets_Bq = {}
        for w in windows:
            wi = w_index[w]
            center_sets_A[w] = set(int(v) for v in centers_of(pack0, 0, wi, 0, 1))
            center_sets_Bq[w] = set(int(v) for v in centers_of(pack0, 0, wi, 1, 0))
        inter_A = set.intersection(*center_sets_A.values())
        # Analysis B quarter grids can be disjoint from half. Segment queries are the common-time fallback.
        covered_A = []
        for t in SEG_CENTERS:
            if all(min(s) <= int(t) <= max(s) for s in center_sets_A.values() if s):
                covered_A.append(int(t))
        coverage_rows.append(
            {
                "grid": grid,
                "analysis": "A",
                "n_intersection_centers": len(inter_A),
                "intersection_centers": json.dumps(sorted(inter_A)),
                "n_segment_centers_covered_by_all_W": len(covered_A),
                "segment_centers_covered": json.dumps(covered_A),
                "per_window_n_A": json.dumps({str(w): len(center_sets_A[w]) for w in windows}),
                "per_window_n_B_quarter": json.dumps({str(w): len(center_sets_Bq[w]) for w in windows}),
                "per_window_n_shared_A_vs_B_quarter": json.dumps(
                    {str(w): len(center_sets_A[w] & center_sets_Bq[w]) for w in windows}
                ),
            }
        )
        for seed in seeds:
            pack = pack0 if int(seed) == int(seeds[0]) else load_windows(out_dir, grid, int(seed))
            for noise_index in noises:
                for w in windows:
                    wi = w_index[w]
                    for analysis, range_code, range_name, allowed, kind in (
                        (0, 0, "quarter", inter_A, "intersection_A"),
                        (0, 1, "half", inter_A, "intersection_A"),
                    ):
                        if len(allowed) == 0:
                            continue
                        rows.append(
                            {
                                "grid": grid,
                                "seed": int(seed),
                                "noise_index": int(noise_index),
                                "window_size": int(w),
                                "analysis": "A",
                                "search_range": range_name,
                                "grid_kind": kind,
                                "n_points": len(allowed),
                                "rmse_ms": subset_rmse_ms(pack, int(noise_index), wi, analysis, range_code, allowed),
                            }
                        )
                    for analysis, range_code, range_name in ((0, 0, "quarter"), (0, 1, "half")):
                        rows.append(
                            {
                                "grid": grid,
                                "seed": int(seed),
                                "noise_index": int(noise_index),
                                "window_size": int(w),
                                "analysis": "A",
                                "search_range": range_name,
                                "grid_kind": "segment_centers_common",
                                "n_points": len(covered_A),
                                "rmse_ms": map_segment_rmse_ms(
                                    pack, int(noise_index), wi, analysis, range_code, covered_A
                                ),
                            }
                        )
    detail = pd.DataFrame(rows)
    summary_rows = []
    if len(detail):
        for key, g in detail.groupby(["grid", "analysis", "grid_kind", "noise_index", "window_size"], sort=True):
            q = g[g.search_range == "quarter"].sort_values("seed")
            h = g[g.search_range == "half"].sort_values("seed")
            diff = h.rmse_ms.to_numpy() - q.rmse_ms.to_numpy()
            ci = mean_ci(diff)
            summary_rows.append(
                {
                    "grid": key[0],
                    "analysis": key[1],
                    "grid_kind": key[2],
                    "noise_index": int(key[3]),
                    "window_size": int(key[4]),
                    "n_points": int(q.n_points.iloc[0]),
                    "mean_rmse_quarter_ms": float(np.nanmean(q.rmse_ms)),
                    "mean_rmse_half_ms": float(np.nanmean(h.rmse_ms)),
                    "mean_diff_ms": ci["mean"],
                    "ci95_low_ms": ci["low"],
                    "ci95_high_ms": ci["high"],
                    "ci_excludes_0": bool(np.isfinite(ci["low"]) and (ci["low"] > 0 or ci["high"] < 0)),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(coverage_rows)


def heatmap(ax, pivot, title, cmap, vmin=None, vmax=None, center=None, fmt="{:.2f}"):
    data = pivot.to_numpy(dtype=float)
    if center is not None:
        lim = np.nanmax(np.abs(data))
        if not np.isfinite(lim) or lim == 0:
            lim = 1.0
        vmin, vmax = -lim, lim
    im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.3g}" for v in pivot.index])
    ax.set_xlabel("W (samples)")
    ax.set_ylabel("noise std")
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isfinite(val):
                ax.text(j, i, fmt.format(val), ha="center", va="center", fontsize=7, color="black")
    return im


def plot_heatmaps(fixed: pd.DataFrame, fig_dir: Path):
    paths = []
    for grid, g0 in fixed.groupby("grid"):
        for analysis, g in g0.groupby("analysis"):
            fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2), constrained_layout=True)
            q = g.pivot(index="noise_std", columns="window_size", values="mean_rmse_quarter_ms")
            h = g.pivot(index="noise_std", columns="window_size", values="mean_rmse_half_ms")
            d = g.pivot(index="noise_std", columns="window_size", values="mean_diff_half_minus_quarter_ms")
            vmax = np.nanpercentile(np.concatenate([q.to_numpy().ravel(), h.to_numpy().ravel()]), 95)
            im0 = heatmap(axes[0], q, "quarter  |t|≤⌊W/4⌋", "viridis", vmin=0, vmax=vmax)
            im1 = heatmap(axes[1], h, "half  |t|≤⌊W/2⌋", "viridis", vmin=0, vmax=vmax)
            im2 = heatmap(axes[2], d, "half − quarter", "RdBu_r", center=0)
            fig.colorbar(im0, ax=axes[:2].tolist(), fraction=0.046, pad=0.04, label="mean RMSE (ms)")
            fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04, label="paired Δ RMSE (ms)")
            fig.suptitle(f"{grid} / analysis {analysis}: seed-mean delay RMSE")
            path = fig_dir / f"heatmap_rmse_{grid}_analysis{analysis}.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(str(path))
    return paths


def plot_curves(choices: pd.DataFrame, fig_dir: Path):
    paths = []
    for (grid, analysis), g in choices.groupby(["grid", "analysis"]):
        fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
        series = [
            ("quarter", "selected_rmse_ms", "quarter quality", "-"),
            ("half", "selected_rmse_ms", "half quality", "-"),
            ("quarter", "oracle_rmse_ms", "quarter oracle", "--"),
            ("half", "oracle_rmse_ms", "half oracle", "--"),
        ]
        for range_name, col, label, ls in series:
            sub = g[g.search_range == range_name]
            xs, ys, lo, hi = [], [], [], []
            for noise, sg in sub.groupby("noise_std"):
                ci = mean_ci(sg[col].to_numpy())
                xs.append(float(noise))
                ys.append(ci["mean"])
                lo.append(ci["low"])
                hi.append(ci["high"])
            order = np.argsort(xs)
            xs, ys, lo, hi = [np.asarray(v)[order] for v in (xs, ys, lo, hi)]
            ax.plot(xs, ys, ls, marker="o", label=label)
            ax.fill_between(xs, lo, hi, alpha=0.15)
        # Baselines do not depend on the NeuroTD search range. Plot once from quarter rows.
        base = g[g.search_range == "quarter"]
        for col, label in (
            ("rmse_dtw_oracle_ms", "DTW oracle min"),
            ("rmse_ctw_oracle_ms", "CTW oracle min"),
            ("rmse_fft_oracle_ms", "FFT oracle min"),
        ):
            xs, ys = [], []
            for noise, sg in base.groupby("noise_std"):
                xs.append(float(noise))
                ys.append(float(sg[col].mean()))
            order = np.argsort(xs)
            ax.plot(np.asarray(xs)[order], np.asarray(ys)[order], ":", marker=".", label=label)
        ax.set_yscale("log")
        ax.set_xlabel("noise std")
        ax.set_ylabel("RMSE (ms)")
        ax.set_title(f"{grid} / analysis {analysis}: quality selection and oracle")
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.legend(fontsize=8, ncol=2)
        path = fig_dir / f"curves_selection_{grid}_analysis{analysis}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
        # selection distribution
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
        for ax, range_name in zip(axes, ("quarter", "half")):
            sub = g[g.search_range == range_name]
            ct = pd.crosstab(sub.window_size if False else sub.selected_W, sub.noise_std)
            # fraction of seeds
            ct = ct.div(ct.sum(axis=0), axis=1)
            im = ax.imshow(ct.to_numpy(), aspect="auto", cmap="Blues", vmin=0, vmax=1)
            ax.set_xticks(range(ct.shape[1]))
            ax.set_xticklabels([f"{v:.3g}" for v in ct.columns], rotation=45, ha="right")
            ax.set_yticks(range(ct.shape[0]))
            ax.set_yticklabels([str(v) for v in ct.index])
            ax.set_xlabel("noise std")
            ax.set_ylabel("selected W")
            ax.set_title(range_name)
            for i in range(ct.shape[0]):
                for j in range(ct.shape[1]):
                    ax.text(j, i, f"{ct.to_numpy()[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=axes, fraction=0.046, label="fraction of seeds")
        fig.suptitle(f"{grid} / analysis {analysis}: quality-selected W")
        path = fig_dir / f"selection_distribution_{grid}_analysis{analysis}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_rates(fixed: pd.DataFrame, fig_dir: Path):
    paths = []
    for (grid, analysis), g in fixed.groupby(["grid", "analysis"]):
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)
        # GT-out is constant across noise; show the first noise row as bars via heatmap anyway.
        for ax, col, title, cmap in (
            (axes[0], "frac_gt_out_quarter", "quarter truth outside radius", "magma"),
            (axes[1], "frac_gt_out_half", "half truth outside radius", "magma"),
            (axes[2], "mean_frac_half_outside_quarter", "half estimate outside quarter radius", "magma"),
        ):
            piv = g.pivot(index="noise_std", columns="window_size", values=col)
            im = heatmap(ax, piv, title, cmap, vmin=0, vmax=1, fmt="{:.2f}")
        fig.colorbar(im, ax=axes, fraction=0.03, label="fraction of windows")
        fig.suptitle(f"{grid} / analysis {analysis}: reachability")
        path = fig_dir / f"rates_reachability_{grid}_analysis{analysis}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
        for ax, col, title in (
            (axes[0], "mean_frac_edge_quarter", "quarter estimate on ±⌊W/4⌋"),
            (axes[1], "mean_frac_edge_half", "half estimate on ±⌊W/2⌋"),
        ):
            piv = g.pivot(index="noise_std", columns="window_size", values=col)
            im = heatmap(ax, piv, title, "cividis", vmin=0, vmax=max(0.05, float(np.nanmax(piv.to_numpy()))), fmt="{:.2f}")
        fig.colorbar(im, ax=axes, fraction=0.046, label="fraction of windows")
        fig.suptitle(f"{grid} / analysis {analysis}: estimates on the search boundary")
        path = fig_dir / f"rates_edge_{grid}_analysis{analysis}.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def plot_traces(out_dir: Path, df: pd.DataFrame, fig_dir: Path):
    paths = []
    seed = 0
    for grid, g in df[df.seed == seed].groupby("grid"):
        windows = sorted(g.window_size.unique())
        chosen_w = [windows[0], windows[len(windows) // 2], windows[-1]]
        noises = sorted(g.noise_std.unique())
        chosen_noise = [noises[0], noises[1], noises[-1]]
        pack = load_windows(out_dir, grid, seed)
        w_index = {int(w): i for i, w in enumerate(windows)}
        noise_values = g.drop_duplicates("noise_index").sort_values("noise_index")
        noise_index_of = {float(r.noise_std): int(r.noise_index) for r in noise_values.itertuples()}
        for noise in chosen_noise:
            ni = noise_index_of[float(noise)]
            fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True, constrained_layout=True)
            for ax, w in zip(axes, chosen_w):
                wi = w_index[int(w)]
                for range_code, name, style in ((0, "quarter", "-"), (1, "half", "--")):
                    m = (
                        (pack["noise_index"] == ni)
                        & (pack["window_index"] == wi)
                        & (pack["analysis"] == 0)
                        & (pack["range_code"] == range_code)
                    )
                    t = pack["center"][m] / FS
                    y = pack["shift"][m].astype(float) * MS
                    ax.plot(t, y, style, label=name, linewidth=1.4)
                mgt = (
                    (pack["noise_index"] == ni)
                    & (pack["window_index"] == wi)
                    & (pack["analysis"] == 0)
                    & (pack["range_code"] == 1)
                )
                ax.plot(pack["center"][mgt] / FS, pack["gt"][mgt].astype(float) * MS, color="black", linewidth=1.2, label="center-mapped truth")
                ax.set_ylabel("delay (ms)")
                ax.set_title(f"W={w}, radius quarter={w // 4}, half={w // 2}")
                ax.grid(True, alpha=0.3)
                ax.legend(loc="upper right", fontsize=8, ncol=3)
            axes[-1].set_xlabel("time (s)")
            fig.suptitle(
                f"{grid} analysis A, seed 0, noise={float(noise):.3g}. "
                "Pre-specified small/middle/large W, not chosen by the result."
            )
            path = fig_dir / f"traces_{grid}_noise{float(noise):.3g}_seed0.png"
            fig.savefig(path, dpi=160)
            plt.close(fig)
            paths.append(str(path))
    return paths


def fmt(x, digits=3):
    if x is None or not np.isfinite(x):
        return "NA"
    return f"{x:.{digits}f}"


def write_report(out_dir: Path, context: dict):
    fixed = context["fixed"]
    sel = context["selection_summary"]
    integrity = context["integrity"]
    env = json.loads((out_dir / "environment.json").read_text())
    params = json.loads((out_dir / "parameter_audit.json").read_text())
    verification = json.loads((out_dir / "verification_seed0.json").read_text())
    completion = json.loads((out_dir / "completion_status.json").read_text())
    lines = []
    a = lines.append
    a("# Discrete-delay search bounds: |t| ≤ ⌊W/4⌋ and |t| ≤ ⌊W/2⌋")
    a("")
    a("This experiment uses the manuscript’s piecewise discrete-delay simulation. The continuous model, iEEG and Patch-seq are outside its scope. Measurements below are distinguished from algebraic bounds and possible mechanisms.")
    a("")
    a("## 1. Findings")
    a("")
    a(context["lead"])
    a("")
    a("## 2. Sources, environment and reproduction")
    a("")
    a("The original audit workspace lacked Git metadata. The archived ZIP identified commit 1a6c4f23818bcfe3152f947c54c553a92edeca0c in its comment. The audit could not verify that commit against local Git objects.")
    a("")
    a(f"- Interpreter: {env['executable']}")
    a(f"- Python: {env['python'].split()[0]}")
    a(f"- NumPy {env['numpy']}, SciPy {env['scipy']}, pandas {env['pandas']}, scikit-learn {env['sklearn']}")
    a(f"- dtw {env['dtw']}, tslearn {env['tslearn']}, scikit-image {env['skimage']}")
    a(f"- Platform: {env['platform']}, {env['machine']}")
    a(f"- Imported NeuroTD source: {env['neurotd_time_varying_shift']}")
    a("")
    a("Source SHA-256 checksums:")
    a("")
    for name, digest in env["source_sha256"].items():
        a(f"- {name}: {digest}")
    a("")
    a("The experiment retained the estimator in src/neurotd. It minimized the weighted MSE separately over each candidate set. The quarter-window estimates were computed directly.")
    a("")
    a("Run from the repository root:")
    a("")
    a("```bash")
    a("cd /path/to/NeuroTD")
    a("python examples/revision/run.py run --workspace revision_runs/range --study search_range --script run_search_range_ablation.py")
    a("```")
    a("")
    a("Completed seeds with an OK checkpoint are skipped. Use --force to recompute them or --analyze-only to rebuild the tables, figures and report from checkpoints.")
    a("")
    a(f"Completion: {completion['expected']} expected grid-by-seed jobs, {len(completion['missing_or_failed'])} missing or failed.")
    if completion["missing_or_failed"]:
        for item in completion["missing_or_failed"]:
            a(f"- Incomplete: {item}")
    else:
        a("Both grids have successful checkpoints for seeds 0–29.")
    a("")
    a("Maximum absolute differences from the September 24 discrete simulation_results_discrete.csv archive for seed 0, the current grid, analysis A and the half-window radius:")
    a("")
    for k, v in verification["prior_csv"]["max_abs"].items():
        a(f"- {k}: {v:.3e}")
    a("")
    a("The comparison covers NeuroTD, DTW, CTW, FFT and E. The seed 0 check compared vectorized half-window centers and integer delays against compute_windowed_mse_with_hop, and the quarter-window implementation against a reference loop.")
    a("")
    a("## 3. Parameter checks")
    a("")
    a("SignalParameters defaults to 512 samples per segment and a sampling rate of 5120 Hz. The segmented signal uses an explicit fs=5120. Millisecond conversions use that explicit rate: 1 sample = 1000/5120 = 0.1953125 ms.")
    a("")
    a(f"Signal length: {params['n_samples']} samples in ten segments of {params['n_samples_per_seg']} samples. The my_ricker width is a = N//16 = {params['ricker_width_a']}. True segment shifts: {params['shifts']}. Maximum true shift: {params['shift_max']} samples. Windows are rectangular, H = W//2 and continuous=False.")
    a("")
    a("The original compute_windowed_mse_with_hop uses max_shift = W//2 for both the search radius and the valid starting positions. Two analyses separate these choices:")
    a("")
    a("- Primary analysis A uses margin = ⌊W/2⌋ for both search ranges. Window centers and data support are held fixed as the set of candidate shifts changes.")
    a("- Sensitivity analysis B uses margin = ⌊W/4⌋ for the quarter radius and ⌊W/2⌋ for the half radius. The half-radius computation is shared between A and B.")
    a("")
    a("Both endpoints are included. The quarter-window range is [−⌊W/4⌋, ⌊W/4⌋] and the half-window range is [−⌊W/2⌋, ⌊W/2⌋]. At tied minima the first scanned (more negative) τ is retained. The stored delay is −τ.")
    a("")
    a("E = sqrt(mean(r_k)), where r_k is the implemented symmetric normalized RMSE in window k. Window selection minimizes E and breaks ties using the smaller W. Manuscript Equation (6), q = 1/(1+sqrt(mean(r_k²))), is recorded only as a diagnostic. Its aggregation differs from E, so maximizing q can yield another window.")
    a("")
    a("For each seed, default_rng(seed) draws two standard normal arrays and scales them by the noise SD. This reproduces the original two generate_noisy calls after resetting the seed for each combination. All window sizes and both search radii therefore use identical x and y within each seed and noise condition. Hashes agreed within each condition.")
    if integrity["issues"]:
        a("")
        a("Input integrity warnings:")
        for item in integrity["issues"]:
            a(f"- {item}")
    a("")
    a("### Simulation grids")
    a("")
    a("The executable grid in fig2_simulation_discrete_shift.py uses W = 33, 65, 129, 257, 513, 1025 and noise levels linspace(0, 0.2, 11). The September 24 discrete reproduction and subsequent S1 comparison used this grid.")
    a("")
    a("The historical grid in the source comments uses W = 51, 119, 171, 257, 513, 1281 and noise levels linspace(0, 0.2, 10). The script identifies this as the setup prior to January 16, 2026. It was fully run and is reported separately. Its use in the submitted manuscript has not been confirmed.")
    a("")
    a("The discrete script identifies these two grids. Figure 3 uses continuous delays. The 40-segment setting belongs to a later diagnostic and is outside this analysis.")
    a("")
    a("### Does the search include the true delay?")
    a("")
    a("The radii and true delays of 8–17 samples determine reachability. A true delay at the inclusive bound is reachable. Window-level out-of-range rates were checked against these bounds.")
    a("")
    a("| Grid | W | ⌊W/4⌋ | ⌊W/2⌋ | True maximum 17 exceeds quarter | Exceeds half | A windows | B quarter windows | Centers shared by A and B quarter |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for grid, spec in params["grids"].items():
        for row in spec["per_window"]:
            a(
                f"| {grid} | {row['window']} | {row['radius_quarter']} | {row['radius_half']} | "
                f"{'yes' if row['quarter_cannot_reach_max_true'] else 'no'} | "
                f"{'yes' if row['half_cannot_reach_max_true'] else 'no'} | "
                f"{row['n_windows_analysis_A']} | {row['n_windows_analysis_B_quarter']} | "
                f"{row['n_shared_A_and_B_quarter_centers']} |"
            )
    a("")
    a("On the current grid the radii for W=33 are 8 and 16 samples. At W=65 the quarter radius is 16. A true shift of 17 therefore falls outside both radii at W=33 and the quarter radius at W=65. On the historical grid W=51 has radii 12 and 25, so only the quarter radius excludes 17. Window-level rates also depend on the segments containing the centers (Section 6).")
    a("")
    a("Quarter-radius analysis B shifts the starting margin to ⌊W/4⌋ while retaining the ⌊W/2⌋ hop. The resulting offset is not an integer multiple of the hop. Shared centers in the table refer to exact intersections of center positions.")
    a("")
    a("## 4. Fixed windows and search radius")
    a("")
    a("The primary comparison is analysis A. For each seed, noise level and W, Δ = RMSE(half) − RMSE(quarter), in ms. Means and paired 95% t intervals use the 30 seeds as independent units. A negative Δ favors the half radius.")
    a("")
    a(context["fixed_narrative"])
    a("")
    a("See tables/fixed_window_paired_ci.csv for condition means and tables/condition_results.csv for seed- and window-level rows.")
    a("")
    a("## 5. Window selection, oracle and cross-window comparisons")
    a("")
    a("For each radius, automatic selection minimizes E without using the true delays. The retrospective oracle minimizes known-delay RMSE on the same candidate set and provides a reference for attainable performance.")
    a("")
    a(context["selection_narrative"])
    a("")
    a("The cross-window comparison evaluates both radii at each of the two selected windows. This separates the effect of the radius at a fixed W from changes in W itself. See tables/cross_window_by_seed.csv.")
    a("")
    a("## 6. Out-of-range truths, boundary hits, local MSE and error bounds")
    a("")
    a("The range-imposed RMSE lower bound is sqrt(mean(max(|τ*| − L, 0)²)), with τ* the delay of the segment nearest the window center. It quantifies the unavoidable error from unreachable delays. The measured RMSE can be larger.")
    a("")
    a(context["bound_narrative"])
    a("")
    a("At shared centers in analysis A, the half-radius set contains all quarter-radius candidates. The minimum local weighted MSE under half should therefore be no larger. The integrity check counts violations. This ordering need not hold for true delay RMSE, E or manuscript q.")
    a("")
    a("## 7. Segment boundaries")
    a("")
    a("As in the original evaluation, each window is assigned the delay of the segment center nearest its center. In a tie argmin selects the earlier segment. A window is marked as mixed when its support crosses a multiple of 512 samples. Its internal delay may vary, but its RMSE is evaluated against the center label.")
    a("")
    a(context["boundary_narrative"])
    a("")
    a("## 8. Common time grids")
    a("")
    a("Primary estimates use the native centers of each W. Two sensitivity analyses evaluate common times without linear interpolation or extrapolation beyond a window’s center range:")
    a("")
    a("- Intersection grid: exact centers shared by all W in analysis A. Estimates and truth are evaluated at those centers.")
    a("- Segment grid: the subset of the ten segment centers within the range covered by every W. The nearest estimated center supplies the delay, with ties resolved to the earlier center. The generating segment supplies the truth.")
    a("")
    a(context["grid_narrative"])
    a("")
    a("## 9. Method rankings")
    a("")
    a("DTW, CTW and FFT use the original SlidingWindowPair configuration and run once per seed, noise level and W. Their constraints are independent of the NeuroTD radius ablation. Figure 2c compares E-selected NeuroTD against each baseline’s minimum RMSE across W. A second comparison evaluates every method at NeuroTD’s selected W. Both appear in tables/ranking_by_seed.csv.")
    a("")
    a(context["rank_narrative"])
    a("")
    a("## 10. Implications for the search-radius formula")
    a("")
    a(context["formula_narrative"])
    a("")
    a("## 11. Outputs")
    a("")
    a("- tables/condition_results.csv: one row per seed, grid, analysis, radius, noise level and W.")
    a("- checkpoints/*_windows.npz: window centers and starts, estimates for both radii, center truth, local minimum MSE and boundary flags. range_code 0/1 denotes quarter/half. analysis 0/1 denotes A/B.")
    a("- tables/fixed_window_paired_ci.csv, selection_by_seed.csv, cross_window_by_seed.csv and ranking_by_seed.csv.")
    a("- figures/: RMSE heat maps, paired differences, window-selection curves and distributions, out-of-range and boundary-hit rates, and prespecified seed 0 trajectories.")
    a("")
    a("The trajectory examples use seed 0. Each grid contributes zero noise, its second noise level and SD 0.2, at its smallest, middle and largest windows. This selection was prespecified.")
    a("")
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rows_where(fixed, grid, analysis):
    return fixed[(fixed.grid == grid) & (fixed.analysis == analysis)].sort_values(["noise_std", "window_size"])


def build_narratives(fixed, choices, sel, cross, rank, grid_sum, coverage) -> dict:
    def sentence_fixed(grid, analysis):
        g = _rows_where(fixed, grid, analysis)
        sig = g[g.ci_excludes_0]
        parts = [f"Analysis {analysis}, grid {grid}: 95% intervals excluded zero in {len(sig)} of {len(g)} noise-by-window cells."]
        if len(sig):
            worse = sig[sig.mean_diff_half_minus_quarter_ms > 0]
            better = sig[sig.mean_diff_half_minus_quarter_ms < 0]
            parts.append(
                f"RMSE was higher with the half radius in {len(worse)} such cells and lower in {len(better)}."
            )
            show = sig.sort_values("mean_diff_half_minus_quarter_ms", key=lambda s: s.abs(), ascending=False).head(8)
            bits = []
            for r in show.itertuples():
                bits.append(
                    f"noise {r.noise_std:.3g}, W={r.window_size}: quarter {r.mean_rmse_quarter_ms:.3f} ms, "
                    f"half {r.mean_rmse_half_ms:.3f} ms, Δ {r.mean_diff_half_minus_quarter_ms:.3f} "
                    f"[{r.ci95_low_ms:.3f}, {r.ci95_high_ms:.3f}], "
                    f"fraction of out-of-range true delays (quarter/half) {r.frac_gt_out_quarter:.3f}/{r.frac_gt_out_half:.3f}, "
                    f"lower bounds {r.lower_bound_quarter_ms:.3f}/{r.lower_bound_half_ms:.3f} ms"
                )
            parts.append("Largest absolute differences: " + "; ".join(bits) + ".")
        unreachable = g[(g.frac_gt_out_quarter > 0) | (g.frac_gt_out_half > 0)]
        reachable = g[(g.frac_gt_out_quarter == 0) & (g.frac_gt_out_half == 0)]
        reach_sig = reachable[reachable.ci_excludes_0]
        parts.append(
            f"Both radii included all true delays in {len(reachable)} cells. Intervals excluded zero in {len(reach_sig)} of these. "
            "Excluding true delays cannot explain differences in those cells."
        )
        if len(unreachable):
            parts.append(
                f"At least one radius excluded some true delays in {len(unreachable)} cells. "
                "The lower bound describes the unavoidable component of the measured error in these cells."
            )
        return " ".join(parts)

    fixed_narrative = "\n\n".join(sentence_fixed(grid, analysis) for grid in fixed.grid.unique() for analysis in ("A", "B"))

    sel_parts = []
    for (grid, analysis), g in sel[sel.search_range != "paired_diff"].groupby(["grid", "analysis"]):
        paired = sel[(sel.grid == grid) & (sel.analysis == analysis) & (sel.search_range == "paired_diff")]
        q = g[g.search_range == "quarter"]
        h = g[g.search_range == "half"]
        bits = []
        for r in q.sort_values("noise_std").itertuples():
            hh = h[h.noise_index == r.noise_index].iloc[0]
            bits.append(
                f"{r.noise_std:.3g}: modal quarter W={r.modal_selected_W} ({r.frac_modal_selected_W:.2f}), "
                f"modal half W={hh.modal_selected_W} ({hh.frac_modal_selected_W:.2f})"
            )
        n_diff_ci = int(paired.quality_ci_excludes_0.sum()) if len(paired) else 0
        mean_same = float(paired.frac_same_selected_W.mean()) if len(paired) else float("nan")
        sel_parts.append(
            f"Grid {grid}, analysis {analysis}. Modal E-selected W across seeds by noise SD: "
            + ", ".join(bits)
            + f". The two radii selected the same W in a mean fraction {mean_same:.3f} of seeds across noise levels. "
            f"Paired post-selection RMSE intervals excluded zero at {n_diff_ci} noise levels. "
            f"E selection agreed with the oracle in a mean fraction {q.frac_quality_equals_oracle_W.mean():.3f} (quarter) "
            f"and {h.frac_quality_equals_oracle_W.mean():.3f} (half). "
            f"The auxiliary manuscript q and E selected different W in a mean fraction {q.frac_paper_q_differs_from_E.mean():.3f} (quarter) "
            f"and {h.frac_paper_q_differs_from_E.mean():.3f} (half). Primary results use E."
        )
    # Cross detail when W differs, analysis A current, averaged already in sel paired rows.
    cross_bits = []
    sub = cross[(cross.analysis == "A") & (~cross.same_selected_W)]
    if len(sub):
        for (grid, noise), g in sub.groupby(["grid", "noise_std"]):
            cross_bits.append(
                f"Grid {grid}, noise SD {float(noise):.3g}: selected windows differed for {len(g)} seeds. "
                f"At the quarter-selected window, mean RMSE was {g.rmse_quarter_at_Wq_ms.mean():.3f} ms (quarter) "
                f"and {g.rmse_half_at_Wq_ms.mean():.3f} ms (half). "
                f"At the half-selected window it was {g.rmse_quarter_at_Wh_ms.mean():.3f} ms (quarter) "
                f"and {g.rmse_half_at_Wh_ms.mean():.3f} ms (half)."
            )
    selection_narrative = "\n\n".join(sel_parts + cross_bits)

    bound_parts = []
    g = fixed[fixed.analysis == "A"]
    for r in g.sort_values(["grid", "window_size", "noise_std"]).itertuples():
        if r.frac_gt_out_quarter == 0 and r.frac_gt_out_half == 0 and r.mean_frac_edge_quarter < 0.01 and r.mean_frac_edge_half < 0.01:
            continue
    # Summarize by window, averaging noise would hide noise dependence of edge rate. Pick noise 0 and max.
    focus = g[g.noise_std.isin(sorted(g.noise_std.unique())[:: max(1, len(g.noise_std.unique()) // 1)])]
    # compact: one line per grid, window at noise min and max
    for (grid, window), gg in g.groupby(["grid", "window_size"]):
        lo = gg.loc[gg.noise_std.idxmin()]
        hi = gg.loc[gg.noise_std.idxmax()]
        if lo.frac_gt_out_quarter == 0 and lo.frac_gt_out_half == 0 and hi.mean_frac_edge_half < 0.02 and hi.mean_frac_edge_quarter < 0.02 and hi.mean_frac_half_outside_quarter < 0.02:
            continue
        bound_parts.append(
            f"Grid {grid}, W={int(window)}. Fraction of windows with unreachable truth (quarter/half): {lo.frac_gt_out_quarter:.3f}/{lo.frac_gt_out_half:.3f}. "
            f"RMSE lower bounds: {lo.lower_bound_quarter_ms:.3f}/{lo.lower_bound_half_ms:.3f} ms. "
            f"At noise SD {lo.noise_std:.3g}, measured RMSE was {lo.mean_rmse_quarter_ms:.3f}/{lo.mean_rmse_half_ms:.3f} ms, "
            f"boundary-hit fractions {lo.mean_frac_edge_quarter:.3f}/{lo.mean_frac_edge_half:.3f}, "
            f"and the fraction of half-radius estimates outside the quarter radius was {lo.mean_frac_half_outside_quarter:.3f}. "
            f"At noise SD {hi.noise_std:.3g}, measured RMSE was {hi.mean_rmse_quarter_ms:.3f}/{hi.mean_rmse_half_ms:.3f} ms, "
            f"boundary-hit fractions {hi.mean_frac_edge_quarter:.3f}/{hi.mean_frac_edge_half:.3f}, "
            f"half-radius estimates outside the quarter radius {hi.mean_frac_half_outside_quarter:.3f}, "
            f"and the fraction at the negative half-radius boundary was {hi.mean_frac_neg_edge_half:.3f}."
        )
    bound_narrative = "\n\n".join(bound_parts) if bound_parts else "Analysis A showed no out-of-range truths or notable boundary hits."

    boundary_parts = []
    ga = fixed[fixed.analysis == "A"]
    for (grid, window), gg in ga.groupby(["grid", "window_size"]):
        r = gg.iloc[0]
        boundary_parts.append(
            f"Grid {grid}, W={int(window)}. Mixed-window fractions (quarter/half): {r.mean_frac_mixed_quarter:.3f}/{r.mean_frac_mixed_half:.3f}. "
            f"Mean fractions of samples whose delay differed from center truth: {r.mean_frac_samples_not_gt_quarter:.3f}/{r.mean_frac_samples_not_gt_half:.3f}. "
            f"Averaged across noise levels, interior-window RMSE was {gg.mean_rmse_interior_quarter_ms.mean():.3f}/{gg.mean_rmse_interior_half_ms.mean():.3f} ms "
            f"and mixed-window RMSE was {gg.mean_rmse_mixed_quarter_ms.mean():.3f}/{gg.mean_rmse_mixed_half_ms.mean():.3f} ms."
        )
    boundary_narrative = "Window positions in analysis A are independent of the search radius. The mixed-window fractions therefore agree.\n\n" + "\n\n".join(boundary_parts)

    grid_parts = []
    for r in coverage.itertuples():
        grid_parts.append(
            f"Grid {r.grid}, analysis A. All W share {r.n_intersection_centers} exact centers, "
            f"and {r.n_segment_centers_covered_by_all_W} segment centers fall within the range of every W. "
            f"Window counts by W: A {r.per_window_n_A}, B quarter {r.per_window_n_B_quarter}. "
            f"Centers shared by A and B quarter: {r.per_window_n_shared_A_vs_B_quarter}."
        )
    if len(grid_sum):
        sig = grid_sum[grid_sum.ci_excludes_0]
        grid_parts.append(
            f"On the common grids, paired RMSE intervals excluded zero in {len(sig)} of {len(grid_sum)} cells. "
            "These are sensitivity checks. The fixed-window table uses native centers for the primary analysis."
        )
        if len(sig):
            top = sig.assign(ad=lambda d: d.mean_diff_ms.abs()).sort_values("ad", ascending=False).head(6)
            bits = [
                f"{r.grid} {r.grid_kind}, noise index {r.noise_index}, W={r.window_size}, Δ {r.mean_diff_ms:.3f} ms"
                for r in top.itertuples()
            ]
            grid_parts.append("Largest absolute differences on shared grids: " + "; ".join(bits) + ".")
    grid_narrative = "\n\n".join(grid_parts)

    rank_parts = []
    for (grid, analysis, range_name), g in rank.groupby(["grid", "analysis", "search_range"]):
        rank_parts.append(
            f"Grid {grid}, analysis {analysis}, {range_name}: the fraction of seed-by-noise pairs in which E-selected NeuroTD had lower RMSE than all three oracle baselines (Figure 2c convention) was "
            f"{g.neurotd_beats_all_oracle_baselines.mean():.3f}. At the same selected W, the fraction was "
            f"{g.neurotd_beats_all_same_window.mean():.3f}. Winner counts under the Figure 2c convention: {g.paper_style_winner.value_counts().to_dict()}. "
            f"Same-window winner counts: {g.same_window_winner.value_counts().to_dict()}."
        )
    # Does quarter change the paper-style winner relative to half?
    change_bits = []
    for (grid, analysis), g in rank.groupby(["grid", "analysis"]):
        q = g[g.search_range == "quarter"][["seed", "noise_index", "paper_style_winner", "same_window_winner", "neurotd_beats_all_oracle_baselines"]]
        h = g[g.search_range == "half"][["seed", "noise_index", "paper_style_winner", "same_window_winner", "neurotd_beats_all_oracle_baselines"]]
        m = q.merge(h, on=["seed", "noise_index"], suffixes=("_q", "_h"))
        change_bits.append(
            f"Grid {grid}, analysis {analysis}. The fraction with a different Figure 2c winner under the two radii was "
            f"{(m.paper_style_winner_q != m.paper_style_winner_h).mean():.3f}. "
            f"Same-window winners differed in a fraction {(m.same_window_winner_q != m.same_window_winner_h).mean():.3f}. "
            f"The fraction in which the conclusion that NeuroTD beat all oracle baselines changed was "
            f"{(m.neurotd_beats_all_oracle_baselines_q != m.neurotd_beats_all_oracle_baselines_h).mean():.3f}."
        )
    rank_narrative = "\n\n".join(rank_parts + change_bits)

    # Formula paragraph uses current analysis A numbers.
    cur = fixed[(fixed.grid == "current") & (fixed.analysis == "A")]
    small = cur[cur.window_size.isin([33, 65])]
    large = cur[cur.window_size == 1025]
    formula = (
        "The implemented bound is |t| ≤ ⌊W/2⌋. On the current grid in analysis A, reducing it to ⌊W/4⌋ mainly affects short windows with unreachable true delays and long windows that admit large aliases."
        f" For W=33 and 65, mean RMSE across the 30 seeds was {small.mean_rmse_quarter_ms.mean():.3f} ms (quarter) and {small.mean_rmse_half_ms.mean():.3f} ms (half). These are simple cell averages."
        f" For W=1025, mean RMSE was {large.mean_rmse_quarter_ms.mean():.3f} ms (quarter) and {large.mean_rmse_half_ms.mean():.3f} ms (half)."
        " Automatic selection avoids some poor fixed windows. It minimizes E, and the resulting frequency of agreement with the known-delay oracle is shown in Section 5."
        " Reducing the radius trades errors from truncated true shifts at small W against errors from accessible aliases at large W."
        " Results for the historical grid are tabulated separately because that grid uses different conditions."
    )
    # Lead from the strongest current-A facts.
    cur_sig = cur[cur.ci_excludes_0]
    n_half_worse = int((cur_sig.mean_diff_half_minus_quarter_ms > 0).sum())
    n_half_better = int((cur_sig.mean_diff_half_minus_quarter_ms < 0).sum())
    lead = (
        f"On the current grid, paired delay RMSE intervals excluded zero in {len(cur_sig)} of {len(cur)} primary fixed-window cells. "
        f"RMSE with the half radius was higher in {n_half_worse} and lower in {n_half_better} of these cells. "
        "At W=33 both radii exclude a true delay of 17 samples. At W=65 only the quarter radius excludes it. For large windows the half radius includes additional distant matches with lower local MSE and potentially larger true delay RMSE. "
        "Minimum-E selection can move away from some small fixed windows. E measures waveform fit and delay RMSE measures delay accuracy. "
        "Section 9 reports how often the E-selected estimate crosses a baseline’s oracle error and changes the Figure 2c ranking across seeds."
    )
    return {
        "lead": lead,
        "fixed_narrative": fixed_narrative,
        "selection_narrative": selection_narrative,
        "bound_narrative": bound_narrative,
        "boundary_narrative": boundary_narrative,
        "grid_narrative": grid_narrative,
        "rank_narrative": rank_narrative,
        "formula_narrative": formula,
    }


def analyze(out_dir: Path | None = None):
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    df = load_summary(out_dir)
    df.to_csv(tab_dir / "condition_results.csv", index=False)
    integrity = check_integrity(df, out_dir)
    (tab_dir / "integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    fixed = fixed_window_table(df)
    fixed.to_csv(tab_dir / "fixed_window_paired_ci.csv", index=False)
    choices, cross, rank = selection_tables(df)
    choices.to_csv(tab_dir / "selection_by_seed.csv", index=False)
    cross.to_csv(tab_dir / "cross_window_by_seed.csv", index=False)
    rank.to_csv(tab_dir / "ranking_by_seed.csv", index=False)
    sel = selection_summary(choices, cross)
    sel.to_csv(tab_dir / "selection_summary.csv", index=False)
    grid_sum, coverage = common_grid_tables(out_dir, df)
    grid_sum.to_csv(tab_dir / "common_grid_sensitivity.csv", index=False)
    coverage.to_csv(tab_dir / "common_grid_coverage.csv", index=False)
    plot_heatmaps(fixed, fig_dir)
    plot_curves(choices, fig_dir)
    plot_rates(fixed, fig_dir)
    plot_traces(out_dir, df, fig_dir)
    narratives = build_narratives(fixed, choices, sel, cross, rank, grid_sum, coverage)
    (tab_dir / "narrative_stats.json").write_text(
        json.dumps({k: v for k, v in narratives.items()}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_report(
        out_dir,
        {
            "fixed": fixed,
            "selection_summary": sel,
            "integrity": integrity,
            **narratives,
        },
    )
    print(f"analysis written to {out_dir}", flush=True)


if __name__ == "__main__":
    analyze()
