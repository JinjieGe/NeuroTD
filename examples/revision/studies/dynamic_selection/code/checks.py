"""Pre-run benchmarks. A float or integer mismatch stops the formal run."""

from __future__ import annotations
from revision_runtime import resolve_path

import json
import os
import time
import traceback

import numpy as np

from science import (
    ALPHA,
    CONFIG_HASH,
    EXPERIMENT_DIR,
    FS,
    N_CAL,
    WINDOWS,
    build_clean_inputs,
    condition_by_id,
    direct_original_window,
    estimate_one_window,
    independent_E,
    load_clean_inputs,
    observe_pair,
    p_value,
    pair_for_record,
    save_clean_inputs,
    select_window,
    statistic_T,
    white_noise,
)

BENCH_DIR = os.path.join(EXPERIMENT_DIR, "benchmarks")
INPUT_PATH = os.path.join(EXPERIMENT_DIR, "inputs", "clean_signals.npz")


def _fail(errors, message):
    errors.append(message)
    print("FAIL", message)


def _ok(message):
    print("OK", message)


def ensure_inputs():
    if os.path.exists(INPUT_PATH):
        clean = load_clean_inputs(INPUT_PATH)
        if clean["_meta"]["config_hash"] != CONFIG_HASH:
            raise RuntimeError("frozen inputs were built from a different scientific config")
        return clean, False
    clean = build_clean_inputs()
    meta = save_clean_inputs(clean, INPUT_PATH)
    clean["_meta"] = meta
    with open(os.path.join(EXPERIMENT_DIR, "inputs", "input_manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return clean, True


def _noisy_pair(x_clean, y_base, seed, sigma):
    rng = np.random.default_rng(seed)
    x = np.asarray(x_clean + white_noise(rng, sigma, len(x_clean)), dtype=np.float64)
    y = np.asarray(y_base + white_noise(rng, sigma, len(y_base)), dtype=np.float64)
    return x, y


def run_fast_tests(clean) -> list[str]:
    errors = []
    # Selection ties follow the fixed window order.
    chosen, tie = select_window([0.4, 0.4, 0.2, 0.2, 0.9, 0.2])
    if chosen != 2 or not tie:
        _fail(errors, f"selection tie expected W=129, got index {chosen} tie {tie}")
    else:
        _ok("selection tie keeps the first minimum")
    chosen, tie = select_window([0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
    if chosen != 0 or tie:
        _fail(errors, f"unique minimum expected index 0, got {chosen} tie {tie}")
    else:
        _ok("unique minimum")
    chosen, _tie = select_window([np.nan, 0.5, 0.5, 0.2, np.inf, 0.2])
    if chosen != 3:
        _fail(errors, f"non-finite E must not win, got {chosen}")
    else:
        _ok("non-finite E does not win")

    # p-value ties are conservative and p is never zero.
    cal = np.zeros(N_CAL)
    if p_value(0.0, cal) != 1.0:
        _fail(errors, "p-value of a tied zero statistic")
    if p_value(1.0, cal) != 0.001:
        _fail(errors, "p-value when the observation exceeds every calibration value")
    cal2 = np.zeros(N_CAL)
    cal2[:49] = 10.0
    if p_value(10.0, cal2) != 0.05:
        _fail(errors, f"tie at the boundary should give p=0.05, got {p_value(10.0, cal2)}")
    if not (p_value(10.0, cal2) < ALPHA):
        _ok("p=0.05 is not called significant")
    else:
        _fail(errors, "p=0.05 was treated as significant")
    cal3 = np.zeros(N_CAL)
    cal3[:48] = 10.0
    if p_value(10.0, cal3) != 0.049:
        _fail(errors, "48 ties should give p=0.049")
    else:
        _ok("48 calibration ties give p=0.049")
    if p_value(float("nan"), cal) == 0 or p_value(float("nan"), cal) < ALPHA:
        _fail(errors, "non-finite observation produced a significant p-value")
    else:
        _ok("non-finite observation is not significant")

    # Constant series has T=0. A known shift of a non-periodic signal is exact.
    if statistic_T(np.full(20, 12)) != 0.0:
        _fail(errors, "T of a constant delay is not 0")
    else:
        _ok("constant delay has T=0")

    t = np.arange(5120, dtype=np.float64)
    x = np.sin(t**1.07) + 0.001 * t
    shift = 5
    y = np.roll(x, shift)
    for model in ("discrete", "continuous"):
        for width in WINDOWS:
            one = estimate_one_window(x, y, width, model)
            if not np.all(one["raw"] == shift):
                _fail(errors, f"known shift {shift} mismatched {model} W={width}")
                break
        else:
            _ok(f"non-periodic roll by {shift} recovered for every {model} window")

    # Same identifiers, same noise bytes. Different identifiers differ.
    a = pair_for_record(clean, 6, 0, 0)[2]
    b = pair_for_record(clean, 6, 0, 0)[2]
    c = pair_for_record(clean, 6, 0, 1)[2]
    if a != b or a == c:
        _fail(errors, "formal noise stream is not a function of the identifiers alone")
    else:
        _ok("formal seeds are identifier-stable")
    return errors


def run_estimator_benchmarks(clean) -> tuple[list[str], dict]:
    errors = []
    rows = []
    cases = (
        ("discrete", clean["x_discrete"], clean["y_discrete_dynamic"], 0),
        ("continuous", clean["x_continuous"], clean["y_continuous_dynamic"], 2025),
    )
    for model, x_clean, y_base, seed in cases:
        for sigma in (0.0, 0.02):
            x, y = _noisy_pair(x_clean, y_base, seed, sigma)
            signal_kind = "dynamic"
            wrapped = observe_pair(x, y, clean, model, signal_kind)
            for one in wrapped["windows"]:
                t0 = time.perf_counter()
                centers, shifts, quality = direct_original_window(x, y, one["window_size"], model)
                direct_sec = time.perf_counter() - t0
                if not np.array_equal(centers, one["centers"]):
                    _fail(errors, f"centers differ {model} W={one['window_size']} sigma={sigma}")
                if not np.array_equal(shifts, one["shifts"]):
                    max_abs = float(np.max(np.abs(np.asarray(shifts, dtype=np.float64) - np.asarray(one["shifts"], dtype=np.float64))))
                    _fail(
                        errors,
                        f"shifts differ {model} W={one['window_size']} sigma={sigma} max_abs={max_abs}",
                    )
                if not np.array_equal(np.float64(quality), np.float64(one["E"])):
                    _fail(
                        errors,
                        f"E differs {model} W={one['window_size']} sigma={sigma}: direct {quality} wrapper {one['E']}",
                    )
                if model == "continuous":
                    rounded = np.round(one["shifts"]).astype(int)
                else:
                    rounded = one["raw"]
                indie = independent_E(x, y, rounded, one["centers"], one["window_size"])
                if not np.array_equal(np.float64(indie), np.float64(one["E"])):
                    _fail(
                        errors,
                        f"independent E differs {model} W={one['window_size']} sigma={sigma}: {indie} vs {one['E']}",
                    )
                rows.append(
                    {
                        "model": model,
                        "sigma": sigma,
                        "window_size": one["window_size"],
                        "E": one["E"],
                        "T": one["T"],
                        "rmse_samples": one["rmse_samples"],
                        "rmse_ms": one["rmse_ms"],
                        "boundary_rate": one["boundary_rate"],
                        "n_windows": one["n_windows"],
                        "direct_sec": direct_sec,
                        "selected_W": WINDOWS[wrapped["selected_index"]],
                    }
                )
            _ok(f"wrapper matches original {model} sigma={sigma} selected W={WINDOWS[wrapped['selected_index']]}")
    if len(rows) != 24:
        _fail(errors, f"expected 24 benchmark rows, got {len(rows)}")
    return errors, {"rows": rows}


def compare_historical(rows) -> dict:
    """Extra check against repository CSVs. A mismatch does not by itself stop the run."""
    import csv

    notes = []
    targets = {
        ("continuous", 0.0): resolve_path('repo', 'simulation_results_continuous.csv'),
        ("continuous", 0.02): resolve_path('repo', 'simulation_results_continuous.csv'),
        ("discrete", 0.0): resolve_path('repo', 'reproduction_audit/full_paper_reproduction_20260925/fig2/simulation_results_discrete.csv'),
        ("discrete", 0.02): resolve_path('repo', 'reproduction_audit/full_paper_reproduction_20260925/fig2/simulation_results_discrete.csv'),
    }
    cache = {}
    comparisons = []
    for (model, sigma), path in targets.items():
        if path not in cache:
            with open(path, newline="") as handle:
                cache[path] = list(csv.DictReader(handle))
        table = cache[path]
        for row in rows:
            if row["model"] != model or float(row["sigma"]) != sigma:
                continue
            match = None
            for old in table:
                if int(float(old["window_size"])) == int(row["window_size"]) and float(old["noise_std"]) == sigma:
                    match = old
                    break
            if match is None:
                notes.append(f"no historical row {model} sigma={sigma} W={row['window_size']}")
                continue
            rmse_diff = abs(float(match["rmse_neurotd"]) - float(row["rmse_samples"]))
            e_diff = abs(float(match["quality_rmse_neurotd"]) - float(row["E"]))
            comparisons.append(
                {
                    "model": model,
                    "sigma": sigma,
                    "window_size": row["window_size"],
                    "rmse_abs_diff": rmse_diff,
                    "E_abs_diff": e_diff,
                    "historical_rmse": float(match["rmse_neurotd"]),
                    "this_rmse": float(row["rmse_samples"]),
                    "historical_E": float(match["quality_rmse_neurotd"]),
                    "this_E": float(row["E"]),
                    "bitwise_rmse": float(match["rmse_neurotd"]) == float(row["rmse_samples"]),
                    "bitwise_E": float(match["quality_rmse_neurotd"]) == float(row["E"]),
                    "source": path,
                }
            )
    n_bit = sum(1 for item in comparisons if item["bitwise_rmse"] and item["bitwise_E"])
    notes.append(
        f"historical CSV comparisons: {n_bit}/{len(comparisons)} bitwise on RMSE and E. "
        "These files are the older dynamic-grid runs, not this Monte Carlo. "
        "A mismatch is recorded and is not used as a substitute checkpoint."
    )
    return {"notes": notes, "comparisons": comparisons}


def run_all_checks():
    os.makedirs(BENCH_DIR, exist_ok=True)
    clean, created = ensure_inputs()
    print(f"inputs {'created' if created else 'reused'} {INPUT_PATH}")
    errors = []
    try:
        errors.extend(run_fast_tests(clean))
        bench_errors, bench = run_estimator_benchmarks(clean)
        errors.extend(bench_errors)
        historical = compare_historical(bench["rows"])
        for note in historical["notes"]:
            print("HIST", note)
    except Exception:
        errors.append(traceback.format_exc())
    report = {
        "passed": not errors,
        "errors": errors,
        "config_hash": CONFIG_HASH,
        "tolerance": {
            "integers": "bitwise",
            "floats": "bitwise",
            "atol": 0.0,
            "rtol": 0.0,
            "established_before_formal_run": True,
            "silent_widening": False,
        },
    }
    if "bench" in locals():
        report["benchmarks"] = bench
    if "historical" in locals():
        report["historical"] = historical
    out = os.path.join(BENCH_DIR, "benchmark_report.json")
    with open(out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print("wrote", out, "passed" if report["passed"] else "FAILED")
    return report


if __name__ == "__main__":
    result = run_all_checks()
    raise SystemExit(0 if result["passed"] else 1)
