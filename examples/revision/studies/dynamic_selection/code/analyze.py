"""Summaries, figures, and the 2.4.3 text. Numbers come from the checkpoints."""

from __future__ import annotations

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import csv
import json
import traceback
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta, binomtest, spearmanr

from science import (
    ALPHA,
    CONDITIONS,
    CONFIG_HASH,
    EXPERIMENT_DIR,
    FIXED_W,
    N_CAL,
    N_DYNAMIC,
    N_NULL,
    PREREGISTERED,
    SPLITS,
    WINDOWS,
    checkpoint_path,
    condition_by_id,
    iter_tasks,
    load_checkpoint,
    load_clean_inputs,
    p_value,
    pair_for_record,
    window_centers_rule,
    hop_for,
    sample_to_time_ms,
    FS,
    truth_for_window,
)

RESULTS = os.path.join(EXPERIMENT_DIR, "results")
FIGURES = os.path.join(EXPERIMENT_DIR, "figures")
REPORTS = os.path.join(EXPERIMENT_DIR, "reports")
FIXED_INDEX = WINDOWS.index(FIXED_W)


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    if n <= 0:
        return float("nan"), float("nan")
    low = 0.0 if k <= 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    high = 1.0 if k >= n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return low, high


def holm(pvals):
    pvals = np.asarray(pvals, dtype=np.float64)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        factor = (m - rank) * pvals[idx]
        running = max(running, factor)
        adjusted[idx] = min(1.0, running)
    return adjusted


def load_rows():
    rows = []
    missing = []
    bad = []
    for task in iter_tasks():
        path = checkpoint_path(*task)
        if not os.path.exists(path):
            missing.append(task)
            continue
        try:
            meta, arrays = load_checkpoint(path)
        except Exception as exc:
            bad.append((task, str(exc)))
            continue
        rows.append(
            {
                "condition_id": int(meta["condition_id"]),
                "split_id": int(meta["split_id"]),
                "replicate_id": int(meta["replicate_id"]),
                "input_hash": meta["input_hash"],
                "config_hash": meta["config_hash"],
                "source_hash": meta["source_hash"],
                "selected_index": int(meta["selected_index"]),
                "selected_W": int(meta["selected_W"]),
                "selection_tie": bool(meta["selection_tie"]),
                "status": meta.get("status"),
                "E": np.asarray(arrays["E"], dtype=np.float64),
                "T": np.asarray(arrays["T"], dtype=np.float64),
                "rmse_samples": np.asarray(arrays["rmse_samples"], dtype=np.float64),
                "rmse_ms": np.asarray(arrays["rmse_ms"], dtype=np.float64),
                "boundary_rate": np.asarray(arrays["boundary_rate"], dtype=np.float64),
                "elapsed_sec": float(meta.get("elapsed_sec", float("nan"))),
            }
        )
    return rows, missing, bad


def _subset(rows, cond, split):
    chosen = [row for row in rows if row["condition_id"] == cond and row["split_id"] == split]
    chosen.sort(key=lambda row: row["replicate_id"])
    return chosen


def _stat(rows, index):
    out = np.empty(len(rows), dtype=np.float64)
    for i, row in enumerate(rows):
        if index == "selected":
            out[i] = row["T"][row["selected_index"]] if row["selected_index"] >= 0 else np.nan
        else:
            out[i] = row["T"][index]
    return out


def _rate(flags):
    flags = np.asarray(flags)
    n = int(flags.size)
    k = int(np.count_nonzero(flags))
    low, high = clopper_pearson(k, n)
    return {"k": k, "n": n, "rate": (k / n) if n else float("nan"), "ci_low": low, "ci_high": high}


def significance(t_obs, t_cal):
    pvals = np.array([p_value(float(value), t_cal) for value in t_obs], dtype=np.float64)
    sig = np.isfinite(pvals) & (pvals < ALPHA)
    return pvals, sig


def summarize(rows):
    by_cond = []
    stability = []
    associations = []
    frequencies = []
    for cond in CONDITIONS:
        cid = cond["condition_id"]
        libraries = {}
        for split in (0, 1, 2):
            cal_rows = _subset(rows, cid, split)
            if len(cal_rows) != N_CAL:
                libraries[split] = None
                continue
            libraries[split] = {
                "selected": _stat(cal_rows, "selected"),
                "fixed": _stat(cal_rows, FIXED_INDEX),
            }
        for split, n_expected, label in (
            (3, N_NULL, "null_test"),
            (4, N_DYNAMIC, "dynamic_test"),
        ):
            test_rows = _subset(rows, cid, split)
            if len(test_rows) != n_expected or libraries.get(0) is None:
                continue
            for mode, index_name in (("selected", "selected"), ("fixed_513", "fixed")):
                t_obs = _stat(test_rows, "selected" if mode == "selected" else FIXED_INDEX)
                library_ps = {}
                for split_id, cal in libraries.items():
                    if cal is None:
                        continue
                    pvals, sig = significance(t_obs, cal[index_name])
                    library_ps[str(split_id)] = {
                        "p": pvals,
                        "rate": _rate(sig),
                        "cutoff_49th_largest": float(np.partition(cal[index_name], -49)[-49]),
                    }
                by_cond.append(
                    {
                        "condition_id": cid,
                        "model": cond["model"],
                        "noise": cond["noise"],
                        "sigma": cond["sigma"],
                        "rho": cond["rho"],
                        "endpoint": label,
                        "mode": mode,
                        "libraries": {
                            key: {
                                "k": value["rate"]["k"],
                                "n": value["rate"]["n"],
                                "rate": value["rate"]["rate"],
                                "ci_low": value["rate"]["ci_low"],
                                "ci_high": value["rate"]["ci_high"],
                                "cutoff_49th_largest": value["cutoff_49th_largest"],
                                "n_nonfinite_p": int(np.count_nonzero(~np.isfinite(value["p"]))),
                            }
                            for key, value in library_ps.items()
                        },
                        "_p": {key: value["p"] for key, value in library_ps.items()},
                        "_sig_main": library_ps["0"]["rate"],
                    }
                )
            # Paired selected versus fixed on the main library.
            if "0" in libraries and libraries[0] is not None:
                t_sel = _stat(test_rows, "selected")
                t_fix = _stat(test_rows, FIXED_INDEX)
                p_sel, sig_sel = significance(t_sel, libraries[0]["selected"])
                p_fix, sig_fix = significance(t_fix, libraries[0]["fixed"])
                rmse_sel = np.array(
                    [
                        row["rmse_ms"][row["selected_index"]] if row["selected_index"] >= 0 else np.nan
                        for row in test_rows
                    ]
                )
                rmse_fix = np.array([row["rmse_ms"][FIXED_INDEX] for row in test_rows])
                e_sel = np.array(
                    [row["E"][row["selected_index"]] if row["selected_index"] >= 0 else np.nan for row in test_rows]
                )
                e_fix = np.array([row["E"][FIXED_INDEX] for row in test_rows])
                stability.append(
                    {
                        "condition_id": cid,
                        "endpoint": label,
                        "n_only_selected": int(np.count_nonzero(sig_sel & ~sig_fix)),
                        "n_only_fixed": int(np.count_nonzero(~sig_sel & sig_fix)),
                        "n_both": int(np.count_nonzero(sig_sel & sig_fix)),
                        "n_neither": int(np.count_nonzero(~sig_sel & ~sig_fix)),
                        "median_rmse_ms_selected_minus_fixed": float(np.nanmedian(rmse_sel - rmse_fix)),
                        "mean_rmse_ms_selected_minus_fixed": float(np.nanmean(rmse_sel - rmse_fix)),
                        "n_selected_rmse_lower": int(np.count_nonzero(rmse_sel < rmse_fix)),
                        "n_selected_rmse_higher": int(np.count_nonzero(rmse_sel > rmse_fix)),
                        "n_selected_rmse_equal": int(np.count_nonzero(rmse_sel == rmse_fix)),
                        "median_E_selected_minus_fixed": float(np.nanmedian(e_sel - e_fix)),
                        "mean_T_selected": float(np.nanmean(t_sel)),
                        "mean_T_fixed": float(np.nanmean(t_fix)),
                    }
                )
            # Agreement of the three libraries on the same test records.
            present = [split_id for split_id, cal in libraries.items() if cal is not None]
            if len(present) >= 2:
                sigs = []
                for split_id in present:
                    _p, sig = significance(
                        _stat(test_rows, "selected"), libraries[split_id]["selected"]
                    )
                    sigs.append(sig)
                stack = np.vstack(sigs)
                stability[-1]["library_ids"] = present
                stability[-1]["n_library_disagreement"] = int(np.count_nonzero(np.ptp(stack, axis=0) > 0))
        # Window frequencies and E-RMSE association on each test split.
        for split, label in ((3, "null_test"), (4, "dynamic_test")):
            test_rows = _subset(rows, cid, split)
            counts = Counter(row["selected_W"] for row in test_rows)
            for width in WINDOWS:
                frequencies.append(
                    {
                        "condition_id": cid,
                        "endpoint": label,
                        "W": width,
                        "n": int(counts.get(width, 0)),
                        "n_total": len(test_rows),
                    }
                )
            spears = []
            matches = 0
            used = 0
            for row in test_rows:
                e_values = row["E"]
                r_values = row["rmse_samples"]
                if np.all(np.isfinite(e_values)) and np.all(np.isfinite(r_values)):
                    used += 1
                    rho = spearmanr(e_values, r_values).statistic
                    if np.isfinite(rho):
                        spears.append(float(rho))
                    if int(np.argmin(e_values)) == int(np.argmin(r_values)):
                        matches += 1
            associations.append(
                {
                    "condition_id": cid,
                    "endpoint": label,
                    "n": used,
                    "median_spearman_E_RMSE": float(np.median(spears)) if spears else float("nan"),
                    "fraction_argmin_E_equals_argmin_RMSE": (matches / used) if used else float("nan"),
                }
            )
    # Holm on the seven main-library null false-positive tests.
    null_selected = [
        item
        for item in by_cond
        if item["endpoint"] == "null_test" and item["mode"] == "selected" and "0" in item["libraries"]
    ]
    null_selected.sort(key=lambda item: item["condition_id"])
    p_raw = []
    for item in null_selected:
        rate = item["libraries"]["0"]
        p_raw.append(float(binomtest(rate["k"], rate["n"], ALPHA, alternative="greater").pvalue))
    p_adj = holm(p_raw) if p_raw else []
    holm_rows = []
    for item, raw, adj in zip(null_selected, p_raw, p_adj):
        holm_rows.append(
            {
                "condition_id": item["condition_id"],
                "k": item["libraries"]["0"]["k"],
                "n": item["libraries"]["0"]["n"],
                "fpr": item["libraries"]["0"]["rate"],
                "p_greater": raw,
                "p_holm": float(adj),
                "holm_significant": bool(adj < ALPHA),
            }
        )
    return {
        "rates": by_cond,
        "paired": stability,
        "frequencies": frequencies,
        "association": associations,
        "holm": holm_rows,
    }


def _public_rates(summary):
    public = []
    for item in summary["rates"]:
        public.append({k: v for k, v in item.items() if not k.startswith("_")})
    return public


def verify_inputs(rows, source_hash):
    clean = load_clean_inputs(os.path.join(EXPERIMENT_DIR, "inputs", "clean_signals.npz"))
    bad_hash = []
    nonfinite = []
    bad_config = []
    ids = []
    for row in rows:
        ids.append((row["condition_id"], row["split_id"], row["replicate_id"]))
        if row["config_hash"] != CONFIG_HASH or row["source_hash"] != source_hash:
            bad_config.append(ids[-1])
        _x, _y, digest = pair_for_record(clean, row["condition_id"], row["split_id"], row["replicate_id"])
        if digest != row["input_hash"]:
            bad_hash.append(ids[-1])
        if not np.all(np.isfinite(row["E"])) or not np.all(np.isfinite(row["T"])) or not np.all(np.isfinite(row["rmse_samples"])):
            nonfinite.append(ids[-1])
    return {
        "n_rows": len(rows),
        "unique_ids": len(set(ids)),
        "duplicate_ids": len(ids) - len(set(ids)),
        "bad_config_or_source": bad_config,
        "bad_input_hash": bad_hash,
        "nonfinite_rows": nonfinite,
        "split_counts": {
            str(split): int(sum(1 for row in rows if row["split_id"] == split)) for split in SPLITS
        },
    }


def audit_sample(stage, source_hash):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import multiprocessing as mp
    from runner import audit_task, init_worker

    reps = PREREGISTERED["audit_reps"]
    splits = [0, 3, 4] if stage == "interim" else [0, 1, 2, 3, 4]
    # Interim checks the first replicate only. The final pass checks first, middle, and last.
    tasks = []
    for cond in CONDITIONS:
        for split in splits:
            chosen = reps[str(split)] if stage == "final" else reps[str(split)][:1]
            for rep in chosen:
                path = checkpoint_path(cond["condition_id"], split, rep)
                if os.path.exists(path):
                    tasks.append((cond["condition_id"], split, rep))
    if not tasks:
        return {"n": 0, "passed": 0, "failures": []}
    ctx = mp.get_context("spawn")
    failures = []
    passed = 0
    with ProcessPoolExecutor(
        max_workers=min(4, len(tasks)), mp_context=ctx, initializer=init_worker, initargs=(source_hash,)
    ) as executor:
        futures = [executor.submit(audit_task, task) for task in tasks]
        for fut in as_completed(futures):
            payload = fut.result()
            if payload["passed"]:
                passed += 1
            else:
                failures.append(payload)
    return {"n": len(tasks), "passed": passed, "failures": failures}


def make_figures(rows, summary):
    os.makedirs(FIGURES, exist_ok=True)
    plt.rcParams.update({"axes.unicode_minus": False, "font.size": 10})
    _fig_rates(summary)
    _fig_stability(summary)
    _fig_frequency(summary)
    _fig_association(rows)
    _fig_curves(rows)
    made = sorted(name for name in os.listdir(FIGURES) if name.endswith(".png"))
    return made


def _fig_rates(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=False)
    for ax, endpoint, title in (
        (axes[0], "null_test", "Null false-positive rate"),
        (axes[1], "dynamic_test", "Dynamic detection rate"),
    ):
        items = [
            item
            for item in summary["rates"]
            if item["endpoint"] == endpoint and item["mode"] in ("selected", "fixed_513") and "0" in item["libraries"]
        ]
        conds = sorted({item["condition_id"] for item in items})
        x = np.arange(len(conds))
        for offset, mode, label in ((-0.15, "selected", "E-selected"), (0.15, "fixed_513", "Fixed W=513")):
            ys, lows, highs = [], [], []
            for cid in conds:
                item = next(row for row in items if row["condition_id"] == cid and row["mode"] == mode)
                lib = item["libraries"]["0"]
                ys.append(lib["rate"])
                lows.append(lib["rate"] - lib["ci_low"])
                highs.append(lib["ci_high"] - lib["rate"])
            ax.errorbar(x + offset, ys, yerr=[lows, highs], fmt="o", capsize=3, label=label)
        ax.axhline(ALPHA, color="gray", lw=1, label="0.05")
        ax.set_xticks(x)
        ax.set_xticklabels([str(cid) for cid in conds])
        ax.set_xlabel("condition")
        ax.set_ylabel("rate")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig1_fpr_detection_ci.png"), dpi=150)
    plt.close(fig)


def _fig_stability(summary):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, endpoint, title in (
        (axes[0], "null_test", "Null FPR across calibration libraries"),
        (axes[1], "dynamic_test", "Detection across calibration libraries"),
    ):
        items = [
            item
            for item in summary["rates"]
            if item["endpoint"] == endpoint and item["mode"] == "selected"
        ]
        conds = sorted({item["condition_id"] for item in items})
        x = np.arange(len(conds))
        for offset, lib, label in ((-0.2, "0", "library 1"), (0.0, "1", "library 2"), (0.2, "2", "library 3")):
            ys = []
            for cid in conds:
                item = next(row for row in items if row["condition_id"] == cid)
                ys.append(item["libraries"].get(lib, {}).get("rate", np.nan))
            ax.plot(x + offset, ys, "o", label=label)
        ax.axhline(ALPHA if endpoint == "null_test" else np.nan, color="gray", lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(cid) for cid in conds])
        ax.set_xlabel("condition")
        ax.set_ylabel("rate")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig2_calibration_libraries.png"), dpi=150)
    plt.close(fig)


def _fig_frequency(summary):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, endpoint, title in (
        (axes[0], "null_test", "Selected window, null test"),
        (axes[1], "dynamic_test", "Selected window, dynamic test"),
    ):
        conds = [cond["condition_id"] for cond in CONDITIONS]
        bottoms = np.zeros(len(conds))
        for width in WINDOWS:
            heights = []
            for cid in conds:
                item = next(
                    row
                    for row in summary["frequencies"]
                    if row["condition_id"] == cid and row["endpoint"] == endpoint and row["W"] == width
                )
                total = item["n_total"] or 1
                heights.append(item["n"] / total)
            ax.bar(np.arange(len(conds)), heights, bottom=bottoms, label=str(width))
            bottoms += np.asarray(heights)
        ax.set_ylabel("frequency")
        ax.set_title(title)
        ax.set_xticks(np.arange(len(conds)))
        ax.set_xticklabels([str(cid) for cid in conds])
    axes[1].set_xlabel("condition")
    axes[0].legend(ncol=6, frameon=False, title="W")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig3_window_frequency.png"), dpi=150)
    plt.close(fig)


def _fig_association(rows):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    colors = plt.cm.viridis(np.linspace(0, 1, len(WINDOWS)))
    for width, color in zip(WINDOWS, colors):
        xs, ys = [], []
        index = WINDOWS.index(width)
        for row in rows:
            if row["split_id"] != 4:
                continue
            xs.append(row["E"][index])
            ys.append(row["rmse_ms"][index])
        ax.scatter(xs, ys, s=8, alpha=0.35, color=color, label=str(width))
    ax.set_xlabel("E")
    ax.set_ylabel("RMSE (ms)")
    ax.set_title("Dynamic test: E and RMSE at each candidate window")
    ax.legend(frameon=False, title="W", markerscale=2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig4_E_vs_RMSE.png"), dpi=150)
    plt.close(fig)


def _fig_curves(rows):
    clean = load_clean_inputs(os.path.join(EXPERIMENT_DIR, "inputs", "clean_signals.npz"))
    wanted = []
    for cond in CONDITIONS:
        for split, reps in (
            (3, PREREGISTERED["curve_replicates"]["null_test_reps"]),
            (4, PREREGISTERED["curve_replicates"]["dynamic_test_reps"]),
        ):
            for rep in reps:
                wanted.append((cond["condition_id"], split, rep))
    fig, axes = plt.subplots(len(CONDITIONS), 2, figsize=(10, 14), sharex=True)
    for cid, cond in enumerate(CONDITIONS):
        for col, split in enumerate((3, 4)):
            ax = axes[cid][col]
            row = next(
                (
                    item
                    for item in rows
                    if item["condition_id"] == cond["condition_id"]
                    and item["split_id"] == split
                    and item["replicate_id"] == 0
                ),
                None,
            )
            if row is None:
                ax.set_visible(False)
                continue
            _plot_one_curve(ax, clean, row)
            ax.set_ylabel(f"c{cond['condition_id']} ms")
            if cid == 0:
                ax.set_title("Null, replicate 0" if split == 3 else "Dynamic, replicate 0")
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig5_preregistered_curves_rep0.png"), dpi=150)
    plt.close(fig)

    # Remaining pre-registered replicates, not chosen by how good they look.
    extra = [rep for rep in (500, 999, 100, 199)]
    fig, axes = plt.subplots(len(CONDITIONS), 4, figsize=(14, 14), sharex=True)
    plan = [(3, 500), (3, 999), (4, 100), (4, 199)]
    for cid, cond in enumerate(CONDITIONS):
        for col, (split, rep) in enumerate(plan):
            ax = axes[cid][col]
            row = next(
                (
                    item
                    for item in rows
                    if item["condition_id"] == cond["condition_id"]
                    and item["split_id"] == split
                    and item["replicate_id"] == rep
                ),
                None,
            )
            if row is None:
                ax.set_visible(False)
                continue
            _plot_one_curve(ax, clean, row)
            if cid == 0:
                ax.set_title(f"{'null' if split == 3 else 'dynamic'} rep {rep}")
            if col == 0:
                ax.set_ylabel(f"c{cond['condition_id']}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "fig5b_preregistered_curves_other_reps.png"), dpi=120)
    plt.close(fig)
    del extra


def _plot_one_curve(ax, clean, row):
    meta_path = checkpoint_path(row["condition_id"], row["split_id"], row["replicate_id"])
    _meta, arrays = load_checkpoint(meta_path)
    spec = condition_by_id(row["condition_id"])
    width = row["selected_W"] if row["selected_W"] in WINDOWS else FIXED_W
    shifts = arrays[f"shifts_{width}"]
    centers = window_centers_rule(len(shifts) and FS * 1 or 5120, width, hop_for(spec["model"], width))
    # centers length must match shifts; regenerate from N_SAMPLES directly
    from science import N_SAMPLES

    centers = window_centers_rule(N_SAMPLES, width, hop_for(spec["model"], width))
    kind = "dynamic" if row["split_id"] == 4 else "null"
    truth = truth_for_window(clean, spec["model"], kind, centers)
    time_s = centers / FS
    ax.plot(time_s, sample_to_time_ms(truth, sampling_rate=FS), color="black", lw=1.2, label="truth")
    ax.plot(time_s, sample_to_time_ms(shifts, sampling_rate=FS), color="#D55E00", lw=1.0, label=f"selected W={width}")
    fixed = arrays[f"shifts_{FIXED_W}"]
    fixed_centers = window_centers_rule(N_SAMPLES, FIXED_W, hop_for(spec["model"], FIXED_W))
    fixed_truth = truth_for_window(clean, spec["model"], kind, fixed_centers)
    ax.plot(
        fixed_centers / FS,
        sample_to_time_ms(fixed, sampling_rate=FS),
        color="#0072B2",
        lw=0.8,
        alpha=0.8,
        label="W=513",
    )
    ax.plot(fixed_centers / FS, sample_to_time_ms(fixed_truth, sampling_rate=FS), color="black", lw=0.6, alpha=0.0)
    ax.legend(frameon=False, fontsize=7)


def write_tables(summary, verification, audit, rows):
    os.makedirs(RESULTS, exist_ok=True)
    rates = _public_rates(summary)
    _write_csv(os.path.join(RESULTS, "rates.csv"), _flatten_rates(rates))
    _write_csv(os.path.join(RESULTS, "paired_selected_vs_fixed.csv"), summary["paired"])
    _write_csv(os.path.join(RESULTS, "window_frequency.csv"), summary["frequencies"])
    _write_csv(os.path.join(RESULTS, "e_rmse_association.csv"), summary["association"])
    _write_csv(os.path.join(RESULTS, "holm_fpr.csv"), summary["holm"])
    record_rows = []
    for row in sorted(rows, key=lambda item: (item["condition_id"], item["split_id"], item["replicate_id"])):
        flat = {
            "condition_id": row["condition_id"],
            "split_id": row["split_id"],
            "replicate_id": row["replicate_id"],
            "selected_W": row["selected_W"],
            "selection_tie": row["selection_tie"],
            "input_hash": row["input_hash"],
        }
        for i, width in enumerate(WINDOWS):
            flat[f"E_{width}"] = row["E"][i]
            flat[f"T_{width}"] = row["T"][i]
            flat[f"rmse_samples_{width}"] = row["rmse_samples"][i]
            flat[f"rmse_ms_{width}"] = row["rmse_ms"][i]
            flat[f"boundary_{width}"] = row["boundary_rate"][i]
        record_rows.append(flat)
    _write_csv(os.path.join(RESULTS, "records_summary.csv"), record_rows)
    payload = {
        "rates": rates,
        "paired": summary["paired"],
        "association": summary["association"],
        "holm": summary["holm"],
        "verification": verification,
        "audit": audit,
    }
    with open(os.path.join(RESULTS, "statistics.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)
        handle.write("\n")
    return payload


def _flatten_rates(rates):
    flat = []
    for item in rates:
        for lib, stats in item["libraries"].items():
            flat.append(
                {
                    "condition_id": item["condition_id"],
                    "model": item["model"],
                    "noise": item["noise"],
                    "sigma": item["sigma"],
                    "rho": item["rho"],
                    "endpoint": item["endpoint"],
                    "mode": item["mode"],
                    "library": lib,
                    **stats,
                }
            )
    return flat


def _write_csv(path, rows):
    rows = list(rows)
    if not rows:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("")
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(type(value))


def _fmt_rate(stats):
    return (
        f"{stats['k']}/{stats['n']} = {stats['rate']:.3f} "
        f"(95% CP {stats['ci_low']:.3f}, {stats['ci_high']:.3f})"
    )


def write_reports(stage, payload, missing, figures):
    os.makedirs(REPORTS, exist_ok=True)
    bench_path = os.path.join(EXPERIMENT_DIR, "benchmarks", "benchmark_report.json")
    with open(bench_path, encoding="utf-8") as handle:
        bench = json.load(handle)
    pilot_path = os.path.join(EXPERIMENT_DIR, "pilot", "decision.json")
    pilot = {}
    if os.path.exists(pilot_path):
        with open(pilot_path, encoding="utf-8") as handle:
            pilot = json.load(handle)
    text = _analysis_report(stage, payload, missing, figures, bench, pilot)
    name = "interim_analysis.md" if stage == "interim" else "analysis.md"
    with open(os.path.join(REPORTS, name), "w", encoding="utf-8") as handle:
        handle.write(text)
    if stage == "final":
        with open(os.path.join(REPORTS, "response_2_4_3_EN.md"), "w", encoding="utf-8") as handle:
            handle.write(_response_en(payload, missing))
        with open(os.path.join(REPORTS, "methods_discussion_suggestions.md"), "w", encoding="utf-8") as handle:
            handle.write(_methods(payload))
    return os.path.join(REPORTS, name)


def _condition_label(cid):
    cond = condition_by_id(cid)
    rho = "" if cond["rho"] is None else f", rho={cond['rho']}"
    return f"{cond['model']}, {cond['noise']} SD={cond['sigma']}{rho}"


def _analysis_report(stage, payload, missing, figures, bench, pilot):
    ver = payload["verification"]
    lines = [
        f"# Selection-aware null simulation for NeuroTD 2.4.3 ({stage})",
        "",
        "This report covers the prespecified simulation. The real-data analysis and manuscript were outside its scope.",
        "All calibration libraries use the same test records. The libraries provide a stability check.",
        "Intervals are conditional on the available calibration libraries.",
        "",
        "## Data integrity",
        "",
        f"- Records read: {ver['n_rows']}. Unique tasks: {ver['unique_ids']}. Duplicate IDs: {ver['duplicate_ids']}.",
        f"- Split counts: {ver['split_counts']}.",
        f"- Missing tasks: {len(missing)}. Configuration or source hash mismatches: {len(ver['bad_config_or_source'])}. Input hash mismatches: {len(ver['bad_input_hash'])}.",
        f"- Retained records with nonfinite E, T or RMSE: {len(ver['nonfinite_rows'])}. Nonfinite p-values do not count as significant.",
        f"- Direct recomputation checks passed: {payload['audit']['passed']}/{payload['audit']['n']}.",
        "",
        "## False positives, detection and the fixed window",
        "",
        "The primary test uses library 1. Its p-value is (1 + number of calibration T values at least as large as the observed T)/1000. Significance requires p < 0.05.",
        "The fixed W=513 result uses its own T from the same six-window run and the corresponding W=513 calibration values.",
        "",
    ]
    for item in payload["rates"]:
        if "0" not in item["libraries"]:
            continue
        if item["mode"] not in ("selected", "fixed_513"):
            continue
        lines.append(
            f"- Condition {item['condition_id']} ({_condition_label(item['condition_id'])}), "
            f"{item['endpoint']}, {item['mode']}: {_fmt_rate(item['libraries']['0'])}."
        )
    lines.extend(["", "## Holm diagnostic for excess false positives in library 1", ""])
    if not payload["holm"]:
        lines.append("The primary-library null tests are incomplete, so this diagnostic was omitted.")
    else:
        for row in payload["holm"]:
            lines.append(
                f"- Condition {row['condition_id']}: FPR={row['fpr']:.3f}, one-sided p={row['p_greater']:.4g}, "
                f"Holm-adjusted p={row['p_holm']:.4g}. Reject a false-positive rate of 0.05: {'yes' if row['holm_significant'] else 'no'}."
            )
        lines.append("Failure to detect excess false positives cannot establish exact calibration.")
    lines.extend(["", "## Calibration libraries", ""])
    for item in payload["rates"]:
        if item["mode"] != "selected":
            continue
        bits = []
        for lib, stats in sorted(item["libraries"].items()):
            bits.append(f"Library {int(lib)+1}: {_fmt_rate(stats)}")
        if bits:
            lines.append(
                f"- Condition {item['condition_id']}, {item['endpoint']}, selected window: " + ", ".join(bits) + "."
            )
    lines.extend(["", "## Paired selected- and fixed-window differences", ""])
    for row in payload["paired"]:
        extra = ""
        if "n_library_disagreement" in row:
            extra = f" Significant decisions differed across the three completed libraries for {row['n_library_disagreement']} records."
        lines.append(
            f"- Condition {row['condition_id']}, {row['endpoint']}: selected only {row['n_only_selected']}, "
            f"fixed only {row['n_only_fixed']}, both {row['n_both']}, neither {row['n_neither']}."
            f" Median paired RMSE difference (selected minus fixed): {row['median_rmse_ms_selected_minus_fixed']:.4g} ms. "
            f"Selected RMSE was lower in {row['n_selected_rmse_lower']}, higher in {row['n_selected_rmse_higher']}, and equal in {row['n_selected_rmse_equal']}."
            + extra
        )
    lines.extend(["", "## E and RMSE", ""])
    for row in payload["association"]:
        lines.append(
            f"- Condition {row['condition_id']}, {row['endpoint']}: median six-window Spearman correlation "
            f"{row['median_spearman_E_RMSE']:.3f}. Minimum E and minimum RMSE selected the same window in a fraction "
            f"{row['fraction_argmin_E_equals_argmin_RMSE']:.3f} (n={row['n']})."
        )
    lines.extend(
        [
            "",
            "The prespecified criterion for consistent detection requires the lower 95% Clopper–Pearson bound for power to exceed 0.5.",
            "Conditions with lower bounds of 0.5 or less do not meet this criterion.",
            "",
            "## Zero-noise benchmark",
            "",
            "Before the full run, wrapper and original outputs matched bit for bit for 24 settings: two models, six windows, zero noise and SD 0.02."
            f" Absolute tolerance for the numerical checks: {bench['tolerance']['atol']}.",
            "The benchmark file records the number of exact RMSE and E matches with the historical dynamic grid CSV. "
            "That CSV covers a separate experiment. The 29,379 Monte Carlo records have their own checkpoints.",
            "",
            "## Parallel execution",
            "",
        ]
    )
    if pilot:
        lines.append(
            f"The pilot selected {pilot.get('chosen_workers')} processes. The estimated speedup over steady-state single-process throughput was "
            f"{pilot.get('speedup_vs_single')}. The sustained-to-short pilot throughput ratio was {pilot.get('sustained_ratio')}. "
            "Pilot split_id=99 is excluded from the formal analysis."
        )
    lines.extend(["", "## Figures", ""])
    for name in figures:
        lines.append(f"- figures/{name}")
    lines.extend(
        [
            "",
            "## Scope and interpretation",
            "",
            "These seven conditions compare minimum-E selection and fixed W=513 for false-positive rates and power. ",
            "They assess agreement across calibration libraries on identical test records and the association between E and RMSE.",
            "Extrapolation to real data, the earlier 2.4.1 hop and reconstruction analyses, and manuscript changes fall outside this experiment.",
            "The p-values depend on 999 null records per library. Ties count in the numerator, so the smallest p-value is positive.",
            "Boundary hits use the original integer search at ±W/2. Continuous-mode output retains the original floating-point overlap-add delay curve.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _response_en(payload, missing):
    return (
        "# Response to reviewer comment 2.4.3\n\n"
        "We ran a simulation-only calibration of the original NeuroTD estimator and its original alignment score E. "
        "There are seven prespecified conditions. Each has three independent null calibration libraries of 999 records, "
        "1000 independent null test records, and 200 independent dynamic test records. "
        "Every record evaluates all six candidate windows and then selects the window with the smallest E. "
        "The fixed W=513 comparison uses that same six-window output and the matching calibration library's W=513 statistics.\n\n"
        "The primary p value uses the first calibration library only: "
        "p=(1+count of calibration statistics greater than or equal to the observed statistic)/1000, "
        "with a threshold of 0.05. The other libraries are used for stability, not for choosing a favorable null. "
        "The 1000 test records are the same records under every library.\n\n"
        "Counts, Clopper-Pearson intervals, Holm-adjusted false-positive diagnostics, window frequencies, "
        "and the E-RMSE comparison are in `results/` and `reports/analysis.md`. "
        f"Missing tasks at the time of this text: {len(missing)}.\n\n"
        "The evidence is limited to these simulation conditions. It is not a real-data calibration, "
        "it does not replace the earlier window, hop, and reconstruction checks, and the manuscript text was not changed. "
        "Absence of a Holm-significant false-positive excess does not prove that the p values are exactly calibrated.\n"
    )


def _methods(payload):
    return (
        "# Suggested methods and discussion text\n\n"
        "The paragraphs below are suggestions for the authors. They were not inserted into the manuscript.\n\n"
        "## Methods\n\n"
        "Window selection was assessed on simulated pairs of length 5120 samples (1 s at 5120 Hz). "
        "Discrete signals used the original segmented Ricker generator and the original piecewise shifts. "
        "Continuous signals used the original reference, clamped cubic delay curve, and continuous delay operator, "
        "with the published generator seed 2025. "
        "The null pair was a constant shift of 12 samples in the discrete model and 0 samples in the continuous model. "
        "Candidate windows were 33, 65, 129, 257, 513, and 1025. "
        "Discrete estimation used a rectangle window and hop W/2. "
        "Continuous estimation used a Hann window, hop 1, and the original overlap-weighted reconstruction. "
        "E was the original sliding-window alignment score. For continuous curves, shifts were rounded to integers only inside that score. "
        "The selected window was the smallest E, with the earliest window kept in case of a tie. "
        "The test statistic was the mean squared deviation of the estimated delay in milliseconds about its own mean, "
        "on the native centers of the selected output. "
        "For each condition, three independent calibration libraries of 999 null records supplied the reference distribution. "
        "The p value was (1 + the number of calibration statistics at least as large as the observed statistic) / 1000. "
        "A separate null test of 1000 records and a dynamic test of 200 records were scored against each library. "
        "The primary decision used the first library. "
        "A fixed window of 513 samples was scored from the same estimates.\n\n"
        "## Discussion\n\n"
        "This calibration describes the prespecified simulations only. "
        "A false-positive rate compatible with 0.05 in these conditions does not establish exact calibration, "
        "and it does not transfer automatically to recorded neural signals. "
        "Where the lower confidence bound on dynamic detection stays at or below one half, "
        "the time-varying delay is not reliably separated from the constant-delay null by this statistic. "
        "Agreement among three calibration libraries reduces, but does not remove, dependence on one finite null sample. "
        "The same 1000 test records are reused across libraries, so the libraries are not 3000 independent tests.\n"
    )


def analyze(stage, source_hash):
    os.makedirs(RESULTS, exist_ok=True)
    rows, missing, bad = load_rows()
    if bad:
        with open(os.path.join(RESULTS, "unreadable_checkpoints.json"), "w", encoding="utf-8") as handle:
            json.dump(bad, handle, indent=2, default=_json_default)
    verification = verify_inputs(rows, source_hash) if rows else {
        "n_rows": 0,
        "unique_ids": 0,
        "duplicate_ids": 0,
        "bad_config_or_source": [],
        "bad_input_hash": [],
        "nonfinite_rows": [],
        "split_counts": {},
    }
    summary = summarize(rows) if rows else {
        "rates": [],
        "paired": [],
        "frequencies": [],
        "association": [],
        "holm": [],
    }
    audit = {"n": 0, "passed": 0, "failures": [], "error": None}
    try:
        if stage in ("interim", "final") and rows:
            audit = audit_sample(stage, source_hash)
    except Exception:
        audit["error"] = traceback.format_exc()
    figures = []
    if rows and stage in ("interim", "final"):
        try:
            figures = make_figures(rows, summary)
        except Exception:
            figures = []
            with open(os.path.join(RESULTS, "figure_error.txt"), "w", encoding="utf-8") as handle:
                handle.write(traceback.format_exc())
    payload = write_tables(summary, verification, audit, rows)
    path = write_reports(stage, payload, missing, figures)
    print("analysis written", path, "rows", len(rows), "missing", len(missing))
    return path


if __name__ == "__main__":
    from science import load_clean_inputs, scientific_source_hash, source_file_hashes

    clean = load_clean_inputs(os.path.join(EXPERIMENT_DIR, "inputs", "clean_signals.npz"))
    source_hash = scientific_source_hash(clean["continuous_source_sha256"], source_file_hashes())
    analyze("final", source_hash)
