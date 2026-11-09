#!/usr/bin/env python3
"""Aggregate the discrete search-range ablation and write tables, figures, and the Chinese report."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from scipy import stats

FS = 5120.0
MS = 1000.0 / FS
SEG_LEN = 512
N_SEGMENTS = 10
SHIFTS = np.arange(8, 18, dtype=int)
SEG_CENTERS = (SEG_LEN // 2 + SEG_LEN * np.arange(N_SEGMENTS)).astype(int)
ANALYSIS_NAME = {0: "A", 1: "B"}
RANGE_NAME = {0: "quarter", 1: "half"}


def setup_font():
    for path in (
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Light.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
    ):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    return "default"


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
    a("# 离散模拟搜索范围消融：|t|≤⌊W/4⌋ 与 |t|≤⌊W/2⌋")
    a("")
    a("本报告只覆盖论文中的分段离散时变延迟模拟。连续延迟、iEEG 和 Patch-seq 不在本次实验内。数字来自本次实际运行；公式能推出的关系单独标明；没有直接证据的机制标为待验证。")
    a("")
    a("## 1. 实际运行发现的结论")
    a("")
    a(context["lead"])
    a("")
    a("## 2. 来源、环境和可重跑入口")
    a("")
    a("工作区没有 `.git` 目录，不能报告当前分支或 `git status`。此前审计记录压缩包注释提交为 `1a6c4f23818bcfe3152f947c54c553a92edeca0c`；本次没有重新验证该提交，因为仓库元数据不在工作区里。")
    a("")
    a(f"- 解释器：`{env['executable']}`")
    a(f"- Python：`{env['python'].split()[0]}`")
    a(f"- NumPy {env['numpy']}，SciPy {env['scipy']}，pandas {env['pandas']}，scikit-learn {env['sklearn']}")
    a(f"- dtw {env['dtw']}，tslearn {env['tslearn']}，scikit-image {env['skimage']}")
    a(f"- 平台：{env['platform']}，{env['machine']}")
    a(f"- NeuroTD 导入文件：`{env['neurotd_time_varying_shift']}`")
    a("")
    a("源文件 SHA256：")
    a("")
    for name, digest in env["source_sha256"].items():
        a(f"- `{name}`：`{digest}`")
    a("")
    a("本次没有修改 `src/neurotd`。消融脚本在候选集合上重新求最小加权 MSE，不把 half 的估计裁剪成分 quarter。")
    a("")
    a("重跑：")
    a("")
    a("```bash")
    a("cd /path/to/NeuroTD")
    a("MPLBACKEND=Agg PYTHONPATH=src \\")
    a("  python \\")
    a("  reproduction_audit/discrete_search_range_ablation_20260926/run_search_range_ablation.py")
    a("```")
    a("")
    a("已完成且状态为 ok 的种子会跳过。`--force` 重算。`--analyze-only` 只根据检查点重做表、图和本报告。")
    a("")
    a(f"完成状态：期望 {completion['expected']} 个网格×种子任务，缺失或失败 {len(completion['missing_or_failed'])} 个。")
    if completion["missing_or_failed"]:
        for item in completion["missing_or_failed"]:
            a(f"- 未完成：{item}")
    else:
        a("两套网格、seed 0–29 全部有 ok 检查点。")
    a("")
    a("seed 0、当前网格、analysis A、half 与 `reproduction_audit_20260924/discrete/simulation_results_discrete.csv` 的最大绝对差：")
    a("")
    for k, v in verification["prior_csv"]["max_abs"].items():
        a(f"- `{k}`：{v:.3e}")
    a("")
    a("该对照同时包含 NeuroTD、DTW、CTW、FFT 和质量分 E。seed 0 还把向量化 half 与 `compute_windowed_mse_with_hop` 的窗口中心和整数延迟逐点比较，并把 quarter 与按原循环改写的参照循环逐点比较。")
    a("")
    a("## 3. 参数核对")
    a("")
    a("`SignalParameters` 的默认 `sample_per_seg` 是 512，因此它自己的 `sampling_rate` 也是 5120 Hz。分段信号另按显式 `fs=5120` 构造。两者数值相同；换算毫秒时脚本传入这个显式 `fs`，不用函数默认参数碰运气。1 sample = 1000/5120 = 0.1953125 ms。")
    a("")
    a(f"信号长度 {params['n_samples']}，10 段，每段 {params['n_samples_per_seg']} 点。波形是 `my_ricker`，宽度参数 a = N//16 = {params['ricker_width_a']}。真实分段延迟是 {params['shifts']}，最大真值 {params['shift_max']} sample。矩形窗，H = W//2，`continuous=False`。")
    a("")
    a("原 `compute_windowed_mse_with_hop` 把 `max_shift = W//2` 同时用于候选延迟和有效窗口起点。本实验拆开：")
    a("")
    a("- 主分析 A：两种范围都使用 half 的窗口起点 `margin = ⌊W/2⌋`。窗口中心和样本支持相同，只改变候选延迟。")
    a("- 补充分析 B：quarter 使用 `margin = ⌊W/4⌋`，half 仍是 `⌊W/2⌋`。half 的 B 与 A 是同一次计算。")
    a("")
    a("候选集合含端点：quarter 为 `[-⌊W/4⌋, ⌊W/4⌋]`，half 为 `[-⌊W/2⌋, ⌊W/2⌋]`。并列最小 MSE 保留扫描中先遇到的 τ，也就是更负的 τ；保存的延迟是 `-τ`。")
    a("")
    a("质量分使用原函数返回值。代码里的 E = sqrt(mean(r_k))，r_k 是逐窗口对称归一化 RMSE，选窗是 E 的 argmin，平局取较小 W。这不是正文公式 (6) 的 q = 1/(1+sqrt(mean(r_k^2)))。q 被存成旁路列，不参与选窗。二者不是同一统计量的单调变换，argmin E 与 argmax q 可以选出不同 W。")
    a("")
    a("同一 seed 只从 `default_rng(seed)` 抽取两段标准正态，再乘以该噪声水平的标准差。这与原脚本每个组合都重置种子后调用两次 `generate_noisy` 一致，因此同一 seed、同一噪声下全部 W 和两种搜索范围共用同一对 x、y。检查没有发现同一条件内 x 或 y 的哈希分裂。")
    if integrity["issues"]:
        a("")
        a("完整性问题：")
        for item in integrity["issues"]:
            a(f"- {item}")
    a("")
    a("### 哪套网格对应当前复现")
    a("")
    a("当前网格 W = 33, 65, 129, 257, 513, 1025，噪声 `linspace(0, 0.2, 11)`，是 `fig2_simulation_discrete_shift.py` 里实际执行的网格。2026-09-24 的离散复现和其后的 S1 对照使用的就是这套，不是注释里的旧网格。")
    a("")
    a("注释中的历史网格 W = 51, 119, 171, 257, 513, 1281，噪声 `linspace(0, 0.2, 10)`，脚本写明是 2026-01-16 之前的 legacy。本次把它完整跑完并分开汇总。没有新的手稿证据把它确认成投稿参数，因此下面不把它叫作已确认的投稿设置。")
    a("")
    a("论文离散模拟没有第三套被脚本写明的条件。`fig3` 是连续延迟。40 段诊断只出现在后来的选窗审计里，不是这篇离散模拟，本次没有并入。")
    a("")
    a("### 真值能否落入搜索半径")
    a("")
    a("这是由延迟 8…17 和半径定义推出的，并与运行得到的窗口级越界比例互相核对。端点包含在内，所以真值等于半径时不算越界。")
    a("")
    a("| 网格 | W | ⌊W/4⌋ | ⌊W/2⌋ | 最大真值 17 超出 quarter | 超出 half | A 的窗口数 | B quarter 窗口数 | A 与 B quarter 的共同中心数 |")
    a("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for grid, spec in params["grids"].items():
        for row in spec["per_window"]:
            a(
                f"| {grid} | {row['window']} | {row['radius_quarter']} | {row['radius_half']} | "
                f"{'是' if row['quarter_cannot_reach_max_true'] else '否'} | "
                f"{'是' if row['half_cannot_reach_max_true'] else '否'} | "
                f"{row['n_windows_analysis_A']} | {row['n_windows_analysis_B_quarter']} | "
                f"{row['n_shared_A_and_B_quarter_centers']} |"
            )
    a("")
    a("当前网格上，W=33 的 quarter 半径是 8，half 是 16；W=65 的 quarter 半径是 16。最大真值 17 因此在 W=33 的两种半径以及 W=65 的 quarter 上不可达。历史网格上，W=51 的 quarter 半径是 12，half 是 25，所以只有 quarter 达不到 17。这些是参数事实。窗口级越界比例还取决于中心落在哪一段，见第 6 节，不等于“10 段里有几段越界”的简单比例。")
    a("")
    a("分析 B 的 quarter 把起点从 ⌊W/2⌋ 改到 ⌊W/4⌋，步长仍是 ⌊W/2⌋。起点差 ⌊W/4⌋ 不是步长的整数倍，所以两套中心网格相位错开。上表的共同中心数是实际交集，不是按数组下标对齐的结果。")
    a("")
    a("## 4. 固定窗口：搜索范围本身")
    a("")
    a("主对比是分析 A。每个 seed、每个噪声、每个 W 上，Δ = RMSE(half) − RMSE(quarter)，单位 ms。均值和 95% t 区间以 seed 为单位，n = 30。重叠窗口没有被当成独立重复。负的 Δ 表示 half 的 RMSE 更小。")
    a("")
    a(context["fixed_narrative"])
    a("")
    a("完整的逐条件均值表在 `tables/fixed_window_paired_ci.csv`。逐 seed、逐 W 的原始行在 `tables/condition_results.csv`。")
    a("")
    a("## 5. 自动选窗、oracle 和交叉窗口")
    a("")
    a("自动选窗对每个搜索范围分别取 E 最小的 W，不使用真实延迟。Oracle 对每个搜索范围分别取真实延迟 RMSE 最小的 W，只作为上界。")
    a("")
    a(context["selection_narrative"])
    a("")
    a("交叉比较在两个被选中的 W 上同时给出两种范围的 RMSE，用来分开“同一 W 上搜索范围的直接作用”和“选中的 W 变了”。明细在 `tables/cross_window_by_seed.csv`。")
    a("")
    a("## 6. 越界、触边、局部 MSE 和下界")
    a("")
    a("范围下界是 sqrt(mean(max(|τ*|−L, 0)^2))，τ* 是原评估使用的窗口中心最近段延迟。它只描述真值不可达所强制的误差，不是方法的实际误差。实际 RMSE 可以高于下界。")
    a("")
    a(context["bound_narrative"])
    a("")
    a("在分析 A 的共同窗口上，half 的候选集合包含 quarter，因此同一加权 MSE 下 half 的最小局部 MSE 不应大于 quarter。本次分析 A 的违反次数见完整性检查。这个关系不保证 half 的真实延迟 RMSE、E 或论文 q 更优。")
    a("")
    a("## 7. 分段边界")
    a("")
    a("原评估保留：每个窗口的真值是离窗口中心最近的段中心上的延迟；中心到两侧段中心等距时，`argmin` 取序号更小的段。窗口样本支持跨过 512 的整数倍时记为边界混合窗。混合窗内部的真实延迟不一定是常数，但主 RMSE 仍用中心真值。")
    a("")
    a(context["boundary_narrative"])
    a("")
    a("## 8. 共同时间网格")
    a("")
    a("不同 W 的原生中心不同。主结果仍用各自的原生中心，以保持和原评估一致。敏感性分析另做两套共同时间点，都不用线性插值，也不向该 W 的中心范围外推：")
    a("")
    a("- 交集中心：分析 A 下所有 W 的中心集合之交。估计和真值都取这些确切中心。")
    a("- 段中心：10 个段中心里、落在每一个 W 的中心范围之内的那些点。延迟取最近窗口中心的估计；等距时取更小的中心。真值取该段中心所属段的分段延迟。")
    a("")
    a(context["grid_narrative"])
    a("")
    a("## 9. 方法排序")
    a("")
    a("DTW、CTW、FFT 用原 `SlidingWindowPair` 配置，每个 seed、噪声和 W 只算一次，搜索约束没有改成 quarter 或 half。图 2c 的画法是：NeuroTD 用质量分选窗，三个基线用各自跨 W 的最小 RMSE。另一套对照把四个方法放在 NeuroTD 所选的同一个 W 上。两种口径都列在 `tables/ranking_by_seed.csv`。")
    a("")
    a(context["rank_narrative"])
    a("")
    a("## 10. 对论文公式的含义")
    a("")
    a(context["formula_narrative"])
    a("")
    a("## 11. 文件")
    a("")
    a("- `tables/condition_results.csv`：全部 seed、网格、分析、范围、噪声和 W 的逐次结果。")
    a("- `checkpoints/*_windows.npz`：逐窗口中心、起点、两种估计、中心真值、局部最小 MSE 和边界标记。`range_code` 0 是 quarter，1 是 half；`analysis` 0 是 A，1 是 B。")
    a("- `tables/fixed_window_paired_ci.csv`、`selection_by_seed.csv`、`cross_window_by_seed.csv`、`ranking_by_seed.csv`。")
    a("- `figures/`：RMSE 热图、配对差、选窗曲线、选窗分布、越界率、触边率和 seed 0 的预指定轨迹。")
    a("")
    a("轨迹使用 seed 0，噪声取该网格的 0、第二档和 0.2，窗口取最小、中间和最大。这是事先规定的代表点，不是按结论挑出来的。")
    a("")
    (out_dir / "REPORT_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _rows_where(fixed, grid, analysis):
    return fixed[(fixed.grid == grid) & (fixed.analysis == analysis)].sort_values(["noise_std", "window_size"])


def build_narratives(fixed, choices, sel, cross, rank, grid_sum, coverage) -> dict:
    def sentence_fixed(grid, analysis):
        g = _rows_where(fixed, grid, analysis)
        sig = g[g.ci_excludes_0]
        parts = [f"分析 {analysis}、网格 `{grid}`：{len(g)} 个噪声×W 单元中，{len(sig)} 个的 95% 区间不含 0。"]
        if len(sig):
            worse = sig[sig.mean_diff_half_minus_quarter_ms > 0]
            better = sig[sig.mean_diff_half_minus_quarter_ms < 0]
            parts.append(
                f"其中 half 显著更高（更差）的有 {len(worse)} 个，half 显著更低（更好）的有 {len(better)} 个。"
            )
            show = sig.sort_values("mean_diff_half_minus_quarter_ms", key=lambda s: s.abs(), ascending=False).head(8)
            bits = []
            for r in show.itertuples():
                bits.append(
                    f"噪声 {r.noise_std:.3g}、W={r.window_size}：quarter {r.mean_rmse_quarter_ms:.3f} ms，"
                    f"half {r.mean_rmse_half_ms:.3f} ms，Δ {r.mean_diff_half_minus_quarter_ms:.3f} "
                    f"[{r.ci95_low_ms:.3f}, {r.ci95_high_ms:.3f}]，"
                    f"真值越界比例 quarter {r.frac_gt_out_quarter:.3f}、half {r.frac_gt_out_half:.3f}，"
                    f"下界 {r.lower_bound_quarter_ms:.3f} / {r.lower_bound_half_ms:.3f} ms"
                )
            parts.append("绝对差最大的单元：" + "；".join(bits) + "。")
        unreachable = g[(g.frac_gt_out_quarter > 0) | (g.frac_gt_out_half > 0)]
        reachable = g[(g.frac_gt_out_quarter == 0) & (g.frac_gt_out_half == 0)]
        reach_sig = reachable[reachable.ci_excludes_0]
        parts.append(
            f"两种范围的真值越界比例都为 0 的单元有 {len(reachable)} 个，其中区间不含 0 的有 {len(reach_sig)} 个。"
            "这些差异不能用“真值超出候选集合”解释。"
        )
        if len(unreachable):
            parts.append(
                f"至少一个范围存在真值越界的单元有 {len(unreachable)} 个。"
                "这些单元里，实际 RMSE 仍应和下界分开看：下界只计入不可达部分。"
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
                f"{r.noise_std:.3g}: quarter 众数 W={r.modal_selected_W}（{r.frac_modal_selected_W:.2f}），"
                f"half 众数 W={hh.modal_selected_W}（{hh.frac_modal_selected_W:.2f}）"
            )
        n_diff_ci = int(paired.quality_ci_excludes_0.sum()) if len(paired) else 0
        mean_same = float(paired.frac_same_selected_W.mean()) if len(paired) else float("nan")
        sel_parts.append(
            f"`{grid}` 分析 {analysis}：各噪声下质量分选窗的 seed 众数为 "
            + "；".join(bits)
            + f"。两种范围选出同一 W 的 seed 比例，对噪声取平均是 {mean_same:.3f}。"
            f"选窗后 RMSE 的配对区间不含 0 的噪声档有 {n_diff_ci} 个。"
            f"质量分所选 W 与 oracle W 相同的 seed 比例，quarter 平均 {q.frac_quality_equals_oracle_W.mean():.3f}，"
            f"half 平均 {h.frac_quality_equals_oracle_W.mean():.3f}。"
            f"旁路公式 q 与 E 选出不同 W 的比例，quarter 平均 {q.frac_paper_q_differs_from_E.mean():.3f}，"
            f"half 平均 {h.frac_paper_q_differs_from_E.mean():.3f}。q 没有用于任何主结果。"
        )
    # Cross detail when W differs, analysis A current, averaged already in sel paired rows.
    cross_bits = []
    sub = cross[(cross.analysis == "A") & (~cross.same_selected_W)]
    if len(sub):
        for (grid, noise), g in sub.groupby(["grid", "noise_std"]):
            cross_bits.append(
                f"`{grid}` 噪声 {float(noise):.3g} 有 {len(g)} 个 seed 选窗不同："
                f"在 quarter 的 W 上，quarter RMSE 均值 {g.rmse_quarter_at_Wq_ms.mean():.3f} ms，"
                f"half {g.rmse_half_at_Wq_ms.mean():.3f} ms；"
                f"在 half 的 W 上，quarter {g.rmse_quarter_at_Wh_ms.mean():.3f} ms，"
                f"half {g.rmse_half_at_Wh_ms.mean():.3f} ms。"
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
            f"`{grid}` W={int(window)}：窗口级真值越界比例 quarter {lo.frac_gt_out_quarter:.3f}、half {lo.frac_gt_out_half:.3f}，"
            f"下界 {lo.lower_bound_quarter_ms:.3f} / {lo.lower_bound_half_ms:.3f} ms。"
            f"噪声 {lo.noise_std:.3g} 时实际 RMSE {lo.mean_rmse_quarter_ms:.3f} / {lo.mean_rmse_half_ms:.3f} ms，"
            f"触边比例 {lo.mean_frac_edge_quarter:.3f} / {lo.mean_frac_edge_half:.3f}，"
            f"half 估计落在 quarter 半径外的比例 {lo.mean_frac_half_outside_quarter:.3f}。"
            f"噪声 {hi.noise_std:.3g} 时实际 RMSE {hi.mean_rmse_quarter_ms:.3f} / {hi.mean_rmse_half_ms:.3f} ms，"
            f"触边 {hi.mean_frac_edge_quarter:.3f} / {hi.mean_frac_edge_half:.3f}，"
            f"half 落在 quarter 外 {hi.mean_frac_half_outside_quarter:.3f}，"
            f"half 负边界比例 {hi.mean_frac_neg_edge_half:.3f}。"
        )
    bound_narrative = "\n\n".join(bound_parts) if bound_parts else "分析 A 中没有记录到真值越界或明显触边。"

    boundary_parts = []
    ga = fixed[fixed.analysis == "A"]
    for (grid, window), gg in ga.groupby(["grid", "window_size"]):
        r = gg.iloc[0]
        boundary_parts.append(
            f"`{grid}` W={int(window)}：混合窗比例 quarter {r.mean_frac_mixed_quarter:.3f}，half {r.mean_frac_mixed_half:.3f}；"
            f"窗口内样本延迟不同于中心真值的平均比例 {r.mean_frac_samples_not_gt_quarter:.3f} / {r.mean_frac_samples_not_gt_half:.3f}。"
            f"噪声平均后，内部 RMSE {gg.mean_rmse_interior_quarter_ms.mean():.3f} / {gg.mean_rmse_interior_half_ms.mean():.3f} ms，"
            f"混合窗 RMSE {gg.mean_rmse_mixed_quarter_ms.mean():.3f} / {gg.mean_rmse_mixed_half_ms.mean():.3f} ms。"
        )
    boundary_narrative = "分析 A 的窗口位置与搜索范围无关，因此两种范围的混合窗比例相同。\n\n" + "\n\n".join(boundary_parts)

    grid_parts = []
    for r in coverage.itertuples():
        grid_parts.append(
            f"`{r.grid}` 分析 A 的全体 W 中心交集有 {r.n_intersection_centers} 个点，"
            f"全体 W 都能覆盖的段中心有 {r.n_segment_centers_covered_by_all_W} 个。"
            f"各 W 的 A 窗口数 {r.per_window_n_A}；B quarter 窗口数 {r.per_window_n_B_quarter}；"
            f"A 与 B quarter 的共同中心数 {r.per_window_n_shared_A_vs_B_quarter}。"
        )
    if len(grid_sum):
        sig = grid_sum[grid_sum.ci_excludes_0]
        grid_parts.append(
            f"共同网格上的配对 RMSE 区间不含 0 的单元有 {len(sig)} / {len(grid_sum)}。"
            "这是敏感性结果。原生中心上的固定窗口表仍是主分析。"
        )
        if len(sig):
            top = sig.assign(ad=lambda d: d.mean_diff_ms.abs()).sort_values("ad", ascending=False).head(6)
            bits = [
                f"{r.grid} {r.grid_kind} 噪声索引 {r.noise_index} W={r.window_size} Δ {r.mean_diff_ms:.3f} ms"
                for r in top.itertuples()
            ]
            grid_parts.append("绝对差较大的共同网格单元：" + "；".join(bits) + "。")
    grid_narrative = "\n\n".join(grid_parts)

    rank_parts = []
    for (grid, analysis, range_name), g in rank.groupby(["grid", "analysis", "search_range"]):
        rank_parts.append(
            f"`{grid}` 分析 {analysis} {range_name}：图 2c 口径下 NeuroTD 质量分 RMSE 小于三个基线 oracle 最小值的 seed×噪声比例是 "
            f"{g.neurotd_beats_all_oracle_baselines.mean():.3f}；同一所选 W 上小于三个基线的比例是 "
            f"{g.neurotd_beats_all_same_window.mean():.3f}。图 2c 口径的获胜者计数：{g.paper_style_winner.value_counts().to_dict()}。"
            f"同一窗口口径：{g.same_window_winner.value_counts().to_dict()}。"
        )
    # Does quarter change the paper-style winner relative to half?
    change_bits = []
    for (grid, analysis), g in rank.groupby(["grid", "analysis"]):
        q = g[g.search_range == "quarter"][["seed", "noise_index", "paper_style_winner", "same_window_winner", "neurotd_beats_all_oracle_baselines"]]
        h = g[g.search_range == "half"][["seed", "noise_index", "paper_style_winner", "same_window_winner", "neurotd_beats_all_oracle_baselines"]]
        m = q.merge(h, on=["seed", "noise_index"], suffixes=("_q", "_h"))
        change_bits.append(
            f"`{grid}` 分析 {analysis}：图 2c 获胜者在 quarter 与 half 之间不同的比例是 "
            f"{(m.paper_style_winner_q != m.paper_style_winner_h).mean():.3f}；"
            f"同一窗口获胜者不同的比例是 {(m.same_window_winner_q != m.same_window_winner_h).mean():.3f}；"
            f"“NeuroTD 优于全部 oracle 基线”这一判断发生改变的比例是 "
            f"{(m.neurotd_beats_all_oracle_baselines_q != m.neurotd_beats_all_oracle_baselines_h).mean():.3f}。"
        )
    rank_narrative = "\n\n".join(rank_parts + change_bits)

    # Formula paragraph uses current analysis A numbers.
    cur = fixed[(fixed.grid == "current") & (fixed.analysis == "A")]
    small = cur[cur.window_size.isin([33, 65])]
    large = cur[cur.window_size == 1025]
    formula = (
        "代码实现的是 |t|≤⌊W/2⌋。正文若改成 |t|≤⌊W/4⌋，在当前网格的主分析 A 上，影响集中在真值不可达的小窗和 half 会选到大幅度别名的大窗。"
        f" W=33 与 W=65 上，30 个 seed 的平均 RMSE：quarter {small.mean_rmse_quarter_ms.mean():.3f} ms，half {small.mean_rmse_half_ms.mean():.3f} ms（这是把这些单元简单平均，不是新的推断单位）。"
        f" W=1025 上 quarter 平均 RMSE {large.mean_rmse_quarter_ms.mean():.3f} ms，half {large.mean_rmse_half_ms.mean():.3f} ms。"
        " 自动选窗会绕开一部分很差的固定窗，但它优化的是 E，不是真实延迟 RMSE；oracle 与质量分选窗不一致的比例已在第 5 节给出。"
        " 因此把公式从 W/2 改成 W/4，并不等于整体提高或降低离散模拟误差，而是把误差从“小窗够不着真值”和“大窗够得着别名”这两类情形之间重新分配。"
        " 历史注释网格的方向是否相同，要看该网格自己的表，不能和第 4 节的当前网格混成一个平均。"
    )
    # Lead from the strongest current-A facts.
    cur_sig = cur[cur.ci_excludes_0]
    n_half_worse = int((cur_sig.mean_diff_half_minus_quarter_ms > 0).sum())
    n_half_better = int((cur_sig.mean_diff_half_minus_quarter_ms < 0).sum())
    lead = (
        f"主分析固定窗口网格、使用当前复现网格时，{len(cur)} 个噪声×W 单元里有 {len(cur_sig)} 个的配对 RMSE 区间不含 0："
        f"half 更差 {n_half_worse} 个，half 更好 {n_half_better} 个。"
        "差异不是均匀地落在所有窗上。小半径把 17 sample 的真值挡在 W=33（两种半径）和 W=65（仅 quarter）之外；大窗上 half 的额外候选包含远离真值的延迟，局部 MSE 可以更小而真实延迟 RMSE 更大。"
        "质量分选窗使用的是代码中的 E 最小化，能够离开一部分固定小窗，但 E 与真实 RMSE 不是同一个目标。"
        "图 2c 那种“NeuroTD 对三个基线的 oracle 最小值”的比较是否改写，取决于选窗后的 RMSE 有没有越过基线，见第 9 节的 seed 比例，而不是某一个示例轨迹。"
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
    setup_font()
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
