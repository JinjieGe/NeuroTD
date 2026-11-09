"""Scientific wrappers for the NeuroTD 2.4.3 selection-null simulation.

Scheduling lives elsewhere. This module only calls the original project
functions and records their outputs. The continuous overlap-add block is a
line-by-line split of ``compute_windowed_mse_with_hop`` so the search runs
once; benchmarks require it to match ``continuous=True`` bitwise.
"""

from __future__ import annotations
from revision_runtime import resolve_path

import ast
import hashlib
import json
import os
import sys

# One thread inside each process. Must precede NumPy and the project imports.
for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = "1"

import numpy as np
from scipy.interpolate import CubicSpline

PROJECT_DIR = resolve_path('repo', '')
EXPERIMENT_DIR = resolve_path('dynamic_selection', '')
SRC_DIR = os.path.join(PROJECT_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from neurotd.neuro_signal import SegmentedSignal, my_ricker
from neurotd.time_varying_shift import (
    SignalParameters,
    compute_shifts_sliding_window_gt,
    compute_windowed_mse_with_hop,
    evaluate_alignment_quality_sliding,
    sample_to_time_ms,
)
from neurotd.utils import rmse

MASTER_SEED = 202610190243
FS = 5120
N_SAMPLES = 5120
DURATION = 1.0
WINDOWS = (33, 65, 129, 257, 513, 1025)
FIXED_W = 513
ALPHA = 0.05
N_CAL = 999
N_NULL = 1000
N_DYNAMIC = 200
P_DENOMINATOR = 1000
WRAPPER_REVISION = 1
NULL_SHIFT_DISCRETE = 12
NULL_SHIFT_CONTINUOUS = 0

# Written before any result is seen. Analysis must follow these rules.
PREREGISTERED = {
    "alpha": ALPHA,
    "reliable_detection_rule": "Clopper-Pearson 95% lower bound > 0.5",
    "fpr_compatible_rule": "0.05 lies inside the Clopper-Pearson 95% interval",
    "inflation_test": "one-sided binomial H1: FPR>0.05, Holm across 7 conditions, main library only",
    "curve_replicates": {
        "null_test_split": 3,
        "null_test_reps": [0, 500, 999],
        "dynamic_test_split": 4,
        "dynamic_test_reps": [0, 100, 199],
    },
    "audit_reps": {
        "0": [0, 499, 998],
        "1": [0, 499, 998],
        "2": [0, 499, 998],
        "3": [0, 500, 999],
        "4": [0, 100, 199],
    },
    "nonfinite_rule": "kept in the denominator; a non-finite p is not counted as significant",
}

CONDITIONS = (
    {"condition_id": 0, "model": "discrete", "noise": "white", "sigma": 0.02, "rho": None},
    {"condition_id": 1, "model": "discrete", "noise": "white", "sigma": 0.10, "rho": None},
    {"condition_id": 2, "model": "discrete", "noise": "white", "sigma": 0.20, "rho": None},
    {"condition_id": 3, "model": "continuous", "noise": "white", "sigma": 0.02, "rho": None},
    {"condition_id": 4, "model": "continuous", "noise": "white", "sigma": 0.10, "rho": None},
    {"condition_id": 5, "model": "continuous", "noise": "white", "sigma": 0.20, "rho": None},
    {"condition_id": 6, "model": "continuous", "noise": "ar1", "sigma": 0.02, "rho": 0.8},
)

SPLITS = {
    0: {"name": "main_calibration", "n": N_CAL, "signal": "null"},
    1: {"name": "calibration_2", "n": N_CAL, "signal": "null"},
    2: {"name": "calibration_3", "n": N_CAL, "signal": "null"},
    3: {"name": "null_test", "n": N_NULL, "signal": "null"},
    4: {"name": "dynamic_test", "n": N_DYNAMIC, "signal": "dynamic"},
}

SCIENCE_SPEC = {
    "wrapper_revision": WRAPPER_REVISION,
    "master_seed": MASTER_SEED,
    "rng": "Generator(PCG64(SeedSequence([master_seed, condition_id, split_id, replicate_id])))",
    "noise_order": "x then y",
    "white": "rng.normal(0, sigma, N)",
    "ar1": [
        "z=rng.normal(0, sigma*sqrt(1-rho*rho), N)",
        "z[0]=rng.normal(0, sigma)",
        "for i in 1..N-1: z[i]+=rho*z[i-1]",
    ],
    "windows": list(WINDOWS),
    "tie_break": "first window in the fixed list",
    "discrete": {
        "window_fun": "rectangle",
        "hop": "W//2",
        "continuous_reconstruction": False,
        "round_shifts_for_E": False,
        "dynamic": "SegmentedSignal.generate_shifted(SignalParameters().shifts)",
        "null": "np.roll(x_clean, 12)",
        "null_truth_samples": NULL_SHIFT_DISCRETE,
        "dynamic_truth": "compute_shifts_sliding_window_gt",
    },
    "continuous": {
        "window_fun": "Hann",
        "hop": 1,
        "continuous_reconstruction": True,
        "round_shifts_for_E": True,
        "clean_rng": "default_rng(2025) then generate_reference_signal, build_delay_curve, apply_continuous_delay",
        "dynamic_truth": "delays[window_centers]*5120",
        "null": "np.roll(x_clean, 0)",
        "null_truth_samples": NULL_SHIFT_CONTINUOUS,
        "Delta0": 0.0,
        "Delta1": 0.01 * 0.2,
        "fDelta": 0.1,
        "spike_jitter_std": 0.001,
        "boundary_mode": "clamped",
        "N_spike": 10,
    },
    "E": "first return of evaluate_alignment_quality_sliding",
    "T": "mean((delay_ms-mean(delay_ms))**2), ddof=0, native centers, no smoothing",
    "delay_ms": "sample_to_time_ms(shifts, sampling_rate=5120)",
    "rmse": "neurotd.utils.rmse on float delays versus truth, in samples",
    "p": "(1+count(T_calibration>=T_observed))/1000",
    "alpha": ALPHA,
    "fixed_W": FIXED_W,
    "boundary_hit": "fraction of raw search shifts with abs(shift)==W//2",
    "preregistered": PREREGISTERED,
}


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_hash(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    header = f"{a.dtype.str}|{a.shape}|".encode("ascii")
    return sha256_bytes(header + a.tobytes())


CONFIG_HASH = sha256_text(canonical_json(SCIENCE_SPEC))


def condition_by_id(condition_id: int) -> dict:
    for row in CONDITIONS:
        if row["condition_id"] == condition_id:
            return row
    raise KeyError(condition_id)


def iter_tasks(splits=None):
    split_ids = list(SPLITS) if splits is None else list(splits)
    for cond in CONDITIONS:
        for split_id in split_ids:
            n = SPLITS[split_id]["n"]
            for rep in range(n):
                yield (cond["condition_id"], split_id, rep)


def expected_task_count(splits=None) -> int:
    return sum(1 for _ in iter_tasks(splits))


def make_rng(condition_id: int, split_id: int, replicate_id: int) -> np.random.Generator:
    entropy = [int(MASTER_SEED), int(condition_id), int(split_id), int(replicate_id)]
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy)))


def white_noise(rng: np.random.Generator, sigma: float, n: int) -> np.ndarray:
    return rng.normal(0.0, float(sigma), int(n))


def ar1_noise(rng: np.random.Generator, sigma: float, rho: float, n: int) -> np.ndarray:
    sigma = float(sigma)
    rho = float(rho)
    n = int(n)
    z = rng.normal(0.0, sigma * np.sqrt(1.0 - rho * rho), n)
    z[0] = rng.normal(0.0, sigma)
    for i in range(1, n):
        z[i] += rho * z[i - 1]
    return z


def draw_noise(rng: np.random.Generator, spec: dict, n: int) -> np.ndarray:
    if spec["noise"] == "white":
        return white_noise(rng, spec["sigma"], n)
    if spec["noise"] == "ar1":
        return ar1_noise(rng, spec["sigma"], spec["rho"], n)
    raise ValueError(spec["noise"])


def window_centers_rule(n_samples: int, window_size: int, hop_size: int) -> np.ndarray:
    max_shift = int(window_size) // 2
    starts = np.arange(max_shift, int(n_samples) - int(window_size) - max_shift + 1, int(hop_size))
    centers = starts + (int(window_size) - 1) / 2.0
    return centers.astype(int)


def hop_for(model: str, window_size: int) -> int:
    if model == "discrete":
        return int(window_size) // 2
    if model == "continuous":
        return 1
    raise ValueError(model)


def window_weights(window_size: int, window_fun: str) -> np.ndarray:
    if window_fun == "rectangle":
        weights = np.ones(window_size, dtype=float)
    elif window_fun == "Hann":
        n = np.arange(window_size)
        weights = 0.5 - 0.5 * np.cos(2 * np.pi * n / (window_size - 1))
    else:
        raise ValueError(window_fun)
    weights /= np.sum(weights)
    return weights


def overlap_add_like_original(raw_shifts, centers, window_size: int, n_samples: int, weights) -> np.ndarray:
    """Same statements as the continuous branch of compute_windowed_mse_with_hop."""
    delay = np.zeros(n_samples, dtype=float)
    weight_accum = np.zeros(n_samples, dtype=float)
    for center, tau in zip(centers.astype(int), raw_shifts):
        start = center - window_size // 2
        end = start + window_size
        if 0 <= start and end <= n_samples:
            delay[start:end] += tau * weights
            weight_accum[start:end] += weights
    valid_mask = weight_accum > 0
    delay[valid_mask] /= weight_accum[valid_mask]
    optimal_shifts = delay[centers.astype(int)]
    return optimal_shifts


def estimate_one_window(x: np.ndarray, y: np.ndarray, window_size: int, model: str):
    hop = hop_for(model, window_size)
    if model == "discrete":
        window_fun = "rectangle"
        continuous = False
    elif model == "continuous":
        window_fun = "Hann"
        continuous = True
    else:
        raise ValueError(model)

    centers, raw = compute_windowed_mse_with_hop(
        x,
        y,
        window_size=int(window_size),
        hop_size=int(hop),
        window_fun=window_fun,
        continuous=False,
    )
    if continuous:
        weights = window_weights(int(window_size), window_fun)
        shifts = overlap_add_like_original(raw, centers, int(window_size), len(x), weights)
        shifts_for_e = np.round(shifts).astype(int)
    else:
        shifts = raw
        shifts_for_e = raw

    quality, _per_window = evaluate_alignment_quality_sliding(
        x, y, shifts_for_e, centers, int(window_size)
    )
    limit = int(window_size) // 2
    n_win = int(raw.shape[0])
    n_boundary = int(np.count_nonzero(np.abs(raw) == limit))
    return {
        "window_size": int(window_size),
        "hop_size": int(hop),
        "centers": np.asarray(centers),
        "shifts": np.asarray(shifts),
        "raw": np.asarray(raw),
        "E": float(quality),
        "n_boundary": n_boundary,
        "n_windows": n_win,
        "boundary_rate": float(n_boundary / n_win) if n_win else float("nan"),
    }


def select_window(e_values) -> tuple[int, bool]:
    """Minimum E. Ties keep the earliest window. Non-finite E never wins."""
    best = None
    tie = False
    for i, value in enumerate(e_values):
        if not np.isfinite(value):
            continue
        if best is None or value < e_values[best]:
            best = i
            tie = False
        elif value == e_values[best]:
            tie = True
    if best is None:
        return -1, False
    return int(best), bool(tie)


def statistic_T(shifts: np.ndarray) -> float:
    delay_ms = np.asarray(sample_to_time_ms(shifts, sampling_rate=FS), dtype=np.float64)
    if delay_ms.size == 0 or not np.all(np.isfinite(delay_ms)):
        return float("nan")
    centered = delay_ms - np.mean(delay_ms)
    return float(np.mean(centered * centered))


def truth_for_window(clean: dict, model: str, signal_kind: str, centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers)
    if model == "discrete" and signal_kind == "dynamic":
        return np.asarray(
            compute_shifts_sliding_window_gt(
                centers, clean["segment_centers"], clean["segment_shifts"]
            )
        )
    if model == "discrete" and signal_kind == "null":
        return np.full(centers.shape, NULL_SHIFT_DISCRETE, dtype=np.float64)
    if model == "continuous" and signal_kind == "dynamic":
        return np.asarray(clean["delays"][centers.astype(int)] * FS, dtype=np.float64)
    if model == "continuous" and signal_kind == "null":
        return np.zeros(centers.shape, dtype=np.float64)
    raise ValueError((model, signal_kind))


def rmse_samples(truth: np.ndarray, shifts: np.ndarray) -> float:
    if truth.shape != shifts.shape or truth.size == 0:
        return float("nan")
    if not np.all(np.isfinite(truth)) or not np.all(np.isfinite(shifts)):
        return float("nan")
    return float(rmse(np.asarray(truth, dtype=np.float64), np.asarray(shifts, dtype=np.float64)))


def p_value(t_observed: float, t_calibration: np.ndarray) -> float:
    cal = np.asarray(t_calibration, dtype=np.float64)
    if cal.shape != (N_CAL,) or not np.isfinite(t_observed) or not np.all(np.isfinite(cal)):
        return float("nan")
    count = int(np.count_nonzero(cal >= t_observed))
    return (1 + count) / float(P_DENOMINATOR)


def load_continuous_api():
    """Compile the original generator functions without executing the script body."""
    path = os.path.join(SRC_DIR, "neurotd", "fig3_simulation_continuous_shift.py")
    source = open(path, "r", encoding="utf-8").read()
    tree = ast.parse(source)
    wanted = {
        "SpikeSpec",
        "make_time",
        "make_spikes",
        "normalize_peak",
        "wrap01",
        "interp1_periodic",
        "generate_reference_signal",
        "build_delay_curve",
        "apply_continuous_delay",
    }
    nodes = []
    lines = source.splitlines(keepends=True)
    chunks = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if name in wanted:
            nodes.append(node)
            chunks.append("".join(lines[node.lineno - 1 : node.end_lineno]))
    missing = wanted.difference(getattr(n, "name", None) for n in nodes)
    if missing:
        raise RuntimeError(f"continuous generators missing: {sorted(missing)}")
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "dataclass": __import__("dataclasses").dataclass,
        "Optional": __import__("typing").Optional,
        "Tuple": __import__("typing").Tuple,
        "np": np,
        "CubicSpline": CubicSpline,
    }
    exec(compile(module, path, "exec"), namespace)
    namespace["__source__"] = "".join(chunks)
    namespace["__source_sha256__"] = sha256_text(namespace["__source__"])
    return namespace


def build_clean_inputs() -> dict:
    params = SignalParameters()
    discrete = SegmentedSignal(
        duration=DURATION, sampling_rate=FS, n_segments=10, shape_func=my_ricker
    )
    x_discrete = np.asarray(discrete.data, dtype=np.float64).copy()
    if x_discrete.shape != (N_SAMPLES,):
        raise RuntimeError(f"discrete length {x_discrete.shape}")
    y_discrete_dynamic = np.asarray(
        discrete.generate_shifted(params.shifts).data, dtype=np.float64
    ).copy()
    y_discrete_null = np.roll(x_discrete, NULL_SHIFT_DISCRETE)
    n_per = int(discrete.n_samples_per_seg)
    starts = n_per * np.arange(discrete.n_segments)
    half = n_per / 2.0
    segment_centers = starts + half
    segment_centers_code = n_per // 2 + n_per * np.arange(discrete.n_segments)
    if not np.array_equal(segment_centers, segment_centers_code):
        raise RuntimeError("segment-center definition disagrees with the original integer formula")

    api = load_continuous_api()
    rng = np.random.default_rng(2025)
    t, x_cont, spike_centers, widths = api["generate_reference_signal"](
        duration=DURATION, sampling_rate=FS, N_spike=10, rng=rng
    )
    _t01, delays = api["build_delay_curve"](
        t=t,
        centers=spike_centers,
        duration=DURATION,
        Delta0=0.0,
        Delta1=0.01 * 0.2,
        fDelta=0.1,
        spike_jitter_std=0.001,
        boundary_mode="clamped",
        rng=rng,
    )
    x_cont = np.asarray(x_cont, dtype=np.float64).copy()
    delays = np.asarray(delays, dtype=np.float64).copy()
    t = np.asarray(t, dtype=np.float64).copy()
    y_cont_dynamic = np.asarray(api["apply_continuous_delay"](t / DURATION, x_cont, delays), dtype=np.float64)
    y_cont_null = np.roll(x_cont, NULL_SHIFT_CONTINUOUS)
    if x_cont.shape != (N_SAMPLES,) or y_cont_dynamic.shape != (N_SAMPLES,):
        raise RuntimeError("continuous length mismatch")

    clean = {
        "x_discrete": x_discrete,
        "y_discrete_dynamic": y_discrete_dynamic,
        "y_discrete_null": np.asarray(y_discrete_null, dtype=np.float64),
        "segment_shifts": np.asarray(params.shifts).copy(),
        "segment_centers": np.asarray(segment_centers_code).copy(),
        "t_continuous": t,
        "x_continuous": x_cont,
        "delays": delays,
        "y_continuous_dynamic": y_cont_dynamic,
        "y_continuous_null": np.asarray(y_cont_null, dtype=np.float64),
        "spike_centers": np.asarray(spike_centers, dtype=np.float64).copy(),
        "spike_widths": np.asarray(widths, dtype=np.float64).copy(),
        "continuous_source_sha256": api["__source_sha256__"],
    }
    return clean


def save_clean_inputs(clean: dict, path: str) -> dict:
    arrays = {k: v for k, v in clean.items() if isinstance(v, np.ndarray)}
    meta = {
        "continuous_source_sha256": clean["continuous_source_sha256"],
        "config_hash": CONFIG_HASH,
        "array_hashes": {k: array_hash(v) for k, v in arrays.items()},
    }
    meta["bundle_hash"] = sha256_text(canonical_json(meta["array_hashes"]))
    payload = dict(arrays)
    payload["meta_json"] = np.frombuffer(
        json.dumps(meta, sort_keys=True).encode("utf-8"), dtype=np.uint8
    ).copy()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "wb") as handle:
        np.savez(handle, **payload)
    os.replace(tmp, path)
    return meta


def load_clean_inputs(path: str) -> dict:
    with np.load(path, allow_pickle=False) as blob:
        clean = {k: blob[k].copy() for k in blob.files if k != "meta_json"}
        meta = json.loads(bytes(blob["meta_json"]).decode("utf-8"))
    clean["continuous_source_sha256"] = meta["continuous_source_sha256"]
    clean["_meta"] = meta
    return clean


def observed_pair(clean: dict, spec: dict, signal_kind: str, rng: np.random.Generator):
    model = spec["model"]
    if model == "discrete":
        x_clean = clean["x_discrete"]
        y_base = clean["y_discrete_dynamic"] if signal_kind == "dynamic" else clean["y_discrete_null"]
    elif model == "continuous":
        x_clean = clean["x_continuous"]
        y_base = clean["y_continuous_dynamic"] if signal_kind == "dynamic" else clean["y_continuous_null"]
    else:
        raise ValueError(model)
    x_noise = draw_noise(rng, spec, N_SAMPLES)
    y_noise = draw_noise(rng, spec, N_SAMPLES)
    x_obs = np.asarray(x_clean + x_noise, dtype=np.float64)
    y_obs = np.asarray(y_base + y_noise, dtype=np.float64)
    return x_obs, y_obs


def pair_for_record(clean: dict, condition_id: int, split_id: int, replicate_id: int):
    spec = condition_by_id(condition_id)
    kind = SPLITS[split_id]["signal"]
    rng = make_rng(condition_id, split_id, replicate_id)
    x_obs, y_obs = observed_pair(clean, spec, kind, rng)
    return x_obs, y_obs, array_hash(np.stack([x_obs, y_obs]))


def observe_pair(x_obs, y_obs, clean, model: str, signal_kind: str) -> dict:
    windows = []
    for width in WINDOWS:
        one = estimate_one_window(x_obs, y_obs, width, model)
        truth = truth_for_window(clean, model, signal_kind, one["centers"])
        one["truth"] = truth
        one["T"] = statistic_T(one["shifts"])
        one["rmse_samples"] = rmse_samples(truth, one["shifts"])
        one["rmse_ms"] = float(sample_to_time_ms(one["rmse_samples"], sampling_rate=FS))
        expected_centers = window_centers_rule(len(x_obs), width, one["hop_size"])
        if not np.array_equal(expected_centers, one["centers"]):
            raise RuntimeError(f"center rule mismatch for W={width}")
        windows.append(one)
    e_values = [row["E"] for row in windows]
    selected, tie = select_window(e_values)
    return {"windows": windows, "selected_index": selected, "selection_tie": tie, "E": e_values}


def independent_E(x, y, shifts, centers, window_size: int) -> float:
    """Recompute E from the stated formula, without calling the project scorer."""
    half_w = int(window_size) // 2
    errors = []
    centers = np.asarray(centers, dtype=int)
    shifts = np.asarray(shifts)
    for center, shift in zip(centers, shifts):
        x_start = int(center) - half_w
        x_end = int(center) + half_w + 1
        y_start = x_start + int(shift)
        y_end = x_end + int(shift)
        if x_start >= 0 and x_end <= len(x) and y_start >= 0 and y_end <= len(y):
            x_win = x[x_start:x_end]
            y_win = y[y_start:y_end]
            mse = np.mean((x_win - y_win) ** 2)
            normalizer = np.sqrt(np.mean(x_win**2) * np.mean(y_win**2)) + 1e-16
            errors.append(np.sqrt(mse) / normalizer)
    if not errors:
        return float("nan")
    return float(np.sqrt(np.mean(np.asarray(errors, dtype=np.float64))))


def direct_original_window(x, y, window_size: int, model: str):
    hop = hop_for(model, window_size)
    window_fun = "rectangle" if model == "discrete" else "Hann"
    continuous = model == "continuous"
    centers, shifts = compute_windowed_mse_with_hop(
        x,
        y,
        window_size=int(window_size),
        hop_size=int(hop),
        window_fun=window_fun,
        continuous=continuous,
    )
    if continuous:
        shifts_for_e = np.round(shifts).astype(int)
    else:
        shifts_for_e = shifts
    quality, _ = evaluate_alignment_quality_sliding(x, y, shifts_for_e, centers, int(window_size))
    return centers, shifts, float(quality)


def source_file_hashes() -> dict:
    found = {}
    for root, _dirs, files in os.walk(PROJECT_DIR):
        if any(part in {".mplconfig", "__pycache__"} for part in root.split(os.sep)):
            continue
        for name in files:
            if name.endswith(".py"):
                path = os.path.join(root, name)
                rel = os.path.relpath(path, PROJECT_DIR)
                with open(path, "rb") as handle:
                    found[rel] = sha256_bytes(handle.read())
    return dict(sorted(found.items()))


def scientific_source_hash(continuous_source_sha: str, file_hashes: dict) -> str:
    keys = [
        "src/neurotd/time_varying_shift.py",
        "src/neurotd/neuro_signal.py",
        "src/neurotd/utils.py",
    ]
    payload = {k: file_hashes[k] for k in keys}
    payload["fig3_generators"] = continuous_source_sha
    return sha256_text(canonical_json(payload))


def checkpoint_path(condition_id: int, split_id: int, replicate_id: int) -> str:
    return os.path.join(
        EXPERIMENT_DIR,
        "checkpoints",
        f"c{condition_id}",
        f"s{split_id}",
        f"r{replicate_id:04d}.npz",
    )


def result_to_arrays(result: dict) -> dict:
    arrays = {
        "E": np.asarray([row["E"] for row in result["windows"]], dtype=np.float64),
        "T": np.asarray([row["T"] for row in result["windows"]], dtype=np.float64),
        "rmse_samples": np.asarray([row["rmse_samples"] for row in result["windows"]], dtype=np.float64),
        "rmse_ms": np.asarray([row["rmse_ms"] for row in result["windows"]], dtype=np.float64),
        "boundary_rate": np.asarray(
            [row["boundary_rate"] for row in result["windows"]], dtype=np.float64
        ),
        "n_boundary": np.asarray([row["n_boundary"] for row in result["windows"]], dtype=np.int64),
        "n_windows": np.asarray([row["n_windows"] for row in result["windows"]], dtype=np.int64),
        "window_sizes": np.asarray(WINDOWS, dtype=np.int64),
    }
    for row in result["windows"]:
        arrays[f"shifts_{row['window_size']}"] = np.asarray(row["shifts"], dtype=np.float64)
        arrays[f"raw_{row['window_size']}"] = np.asarray(row["raw"], dtype=np.int64)
    return arrays


def pack_meta(base: dict) -> np.ndarray:
    return np.frombuffer(json.dumps(base, sort_keys=True).encode("utf-8"), dtype=np.uint8).copy()


def atomic_savez(path: str, arrays: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "wb") as handle:
        np.savez(handle, **arrays)
    os.replace(tmp, path)


def load_checkpoint(path: str) -> tuple[dict, dict]:
    with np.load(path, allow_pickle=False) as blob:
        meta = json.loads(bytes(blob["meta_json"]).decode("utf-8"))
        arrays = {k: blob[k].copy() for k in blob.files if k != "meta_json"}
    return meta, arrays


def stored_result_is_consistent(meta: dict, arrays: dict, clean: dict, source_hash: str) -> tuple[bool, str]:
    """Validate a finished checkpoint against regenerated inputs and stored delays."""
    try:
        cond = int(meta["condition_id"])
        split = int(meta["split_id"])
        rep = int(meta["replicate_id"])
        if meta.get("status") != "ok":
            return False, "status"
        if meta.get("config_hash") != CONFIG_HASH or meta.get("source_hash") != source_hash:
            return False, "hash"
        if int(meta.get("wrapper_revision", -1)) != WRAPPER_REVISION:
            return False, "revision"
        spec = condition_by_id(cond)
        _x, _y, pair_hash = pair_for_record(clean, cond, split, rep)
        if meta.get("input_hash") != pair_hash:
            return False, "input_hash"
        if not np.array_equal(arrays["window_sizes"], np.asarray(WINDOWS, dtype=np.int64)):
            return False, "windows"
        e_values = np.asarray(arrays["E"], dtype=np.float64)
        if e_values.shape != (len(WINDOWS),):
            return False, "E_shape"
        x_obs, y_obs, _ = pair_for_record(clean, cond, split, rep)
        kind = SPLITS[split]["signal"]
        for i, width in enumerate(WINDOWS):
            shifts = np.asarray(arrays[f"shifts_{width}"])
            raw = np.asarray(arrays[f"raw_{width}"])
            if shifts.dtype != np.float64 or raw.dtype != np.int64:
                return False, f"dtype_{width}"
            if shifts.shape != raw.shape or shifts.size != int(arrays["n_windows"][i]):
                return False, f"shape_{width}"
            if not np.array_equal(np.float64(statistic_T(shifts)), np.float64(arrays["T"][i])):
                return False, f"T_{width}"
            limit = int(width) // 2
            rate = float(np.count_nonzero(np.abs(raw) == limit) / raw.size)
            if not np.array_equal(np.float64(rate), np.float64(arrays["boundary_rate"][i])):
                return False, f"boundary_{width}"
            centers = window_centers_rule(N_SAMPLES, width, hop_for(spec["model"], width))
            if spec["model"] == "continuous":
                shifts_for_e = np.round(shifts).astype(int)
            else:
                shifts_for_e = raw
            quality, _ = evaluate_alignment_quality_sliding(
                x_obs, y_obs, shifts_for_e, centers, int(width)
            )
            if not np.array_equal(np.float64(quality), np.float64(e_values[i])):
                return False, f"E_{width}"
            truth = truth_for_window(clean, spec["model"], kind, centers)
            fresh_rmse = rmse_samples(truth, shifts)
            if not np.array_equal(np.float64(fresh_rmse), np.float64(arrays["rmse_samples"][i])):
                return False, f"rmse_{width}"
        selected, tie = select_window(e_values)
        if int(meta["selected_index"]) != selected or bool(meta["selection_tie"]) != tie:
            return False, "selection"
        expected_w = int(WINDOWS[selected]) if selected >= 0 else -1
        if int(meta["selected_W"]) != expected_w:
            return False, "selected_W"
    except Exception as exc:
        return False, f"exception:{type(exc).__name__}:{exc}"
    return True, "ok"


def science_token(meta: dict, arrays: dict) -> str:
    """Hash scientific fields. Runtime metadata is excluded."""
    pieces = [canonical_json({k: meta[k] for k in (
        "condition_id",
        "split_id",
        "replicate_id",
        "input_hash",
        "config_hash",
        "source_hash",
        "selected_index",
        "selection_tie",
        "selected_W",
    )})]
    for key in sorted(arrays):
        if key == "meta_json":
            continue
        pieces.append(key + ":" + array_hash(arrays[key]))
    return sha256_text("\n".join(pieces))
