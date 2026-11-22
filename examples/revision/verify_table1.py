"""Verify the fixed-window RMSE comparison and the S1/S2 grid rankings.

Usage (with the NeuroTD project dependencies installed):
    python verify_table1.py ../NeuroTD

The argument is a checkout containing examples/revision/inputs/baseline_tables.
Both fixed examples are recomputed directly from the seeded signals, including
all four estimators. The 132-row ranking check reads the archived benchmark
errors. Results are printed in milliseconds. No input files are modified.
"""

import argparse
import ast
import csv
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

FS = 5120
W = 513
SD = 0.02
METHODS = ("NeuroTD", "DTW", "CTW", "FFT")
WINDOWS = [33, 65, 129, 257, 513, 1025]


def read_grid(repo, model):
    path = repo / "examples" / "revision" / "inputs" / "baseline_tables" / f"simulation_results_{model}.csv"
    with path.open(newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    rows.sort(key=lambda row: (float(row["noise_std"]), int(row["window_size"])))
    settings = [(float(row["noise_std"]), int(row["window_size"])) for row in rows]
    expected = [(sd, w) for sd in np.linspace(0, 0.2, 11) for w in WINDOWS]
    np.testing.assert_allclose(settings, expected, rtol=0, atol=1e-12)
    errors = np.array([[float(row["rmse_" + name.lower()]) for name in METHODS] for row in rows])
    scores = np.array([float(row["quality_rmse_neurotd"]) for row in rows])
    return errors.reshape(11, 6, 4) * (1000 / FS), scores.reshape(11, 6)


def check_rankings(model, errors, scores):
    # Each minimum compares methods at the same noise level and nominal window.
    ordered = np.sort(errors, axis=2)
    if np.any(np.isclose(ordered[:, :, 0], ordered[:, :, 1], rtol=0, atol=1e-12)):
        raise ValueError("Minimum-RMSE tie in benchmark grid; inspect before counting.")
    wins = errors.argmin(axis=2) == 0
    selected = errors[np.arange(11), scores.argmin(axis=1), 0]
    oracle_baselines = errors[:, :, 1:].min(axis=1)
    counts = (int(wins.sum()), int(wins[:, WINDOWS.index(W)].sum()),
              int(np.all(selected[:, None] < oracle_baselines, axis=1).sum()))
    expected = {"discrete": (19, 11, 11), "continuous": (24, 10, 10)}[model]
    if counts != expected:
        raise ValueError(f"{model}: expected {expected} for S1/S2 but found {counts}")
    print(f"{model}: lowest RMSE at {counts[0]}/66 equal-window settings, "
          f"{counts[1]}/11 at W={W}; minimum-E choice below all baseline oracle errors at {counts[2]}/11.")


def continuous_generators(repo):
    # The Figure 3 file runs the complete benchmark on import. Load its original
    # definitions without running the top-level plotting and parameter sweep.
    path = repo / "examples" / "simulations" / "fig3_simulation_continuous_shift.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    module = ModuleType("_neurotd_figure3_definitions")
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


def simulate(repo, model):
    from neurotd.neuro_signal import SegmentedSignal, my_ricker
    from neurotd.time_varying_shift import SignalParameters, compute_shifts_sliding_window_gt

    if model == "discrete":
        seed = np.random.default_rng(0)
        shifts = SignalParameters().shifts
        original = SegmentedSignal(duration=1.0, sampling_rate=FS, n_segments=10, shape_func=my_ricker)
        shifted = original.generate_shifted(shifts)
        x = original.generate_noisy(seed, SD).data
        y = shifted.generate_noisy(seed, SD).data
        spacing = original.n_samples_per_seg
        segment_centers = spacing // 2 + spacing * np.arange(original.n_segments)
        truth = compute_shifts_sliding_window_gt(np.arange(len(x)), segment_centers, shifts)
        return x, y, truth

    api = continuous_generators(repo)
    rng = np.random.default_rng(2025)
    t, x, centers, _ = api.generate_reference_signal(duration=1.0, sampling_rate=FS, N_spike=10, rng=rng)
    _, seconds = api.build_delay_curve(t=t, centers=centers, duration=1.0, Delta0=0., Delta1=.002,
                                       fDelta=.1, spike_jitter_std=.001, boundary_mode="clamped", rng=rng)
    y = api.apply_continuous_delay(t, x, seconds)
    rng = np.random.default_rng(2025)
    return x + rng.normal(0, SD, x.size), y + rng.normal(0, SD, y.size), seconds * FS


def check_fixed_example(repo, model, archived):
    from neurotd.time_varying_shift import CTWShift, DTWShift, FFTShift, compute_windowed_mse_with_hop
    from neurotd.utils import SlidingWindowPair

    x, y, truth = simulate(repo, model)
    continuous = model == "continuous"
    hop = 1 if continuous else W // 2
    print(f"Recomputing {model} shifts (W={W}, noise SD={SD}) ...", flush=True)
    neuro_centers, neuro = compute_windowed_mse_with_hop(
        x, y, W, hop, window_fun="Hann" if continuous else "rectangle", continuous=continuous)
    windows = SlidingWindowPair(x, y, W, hop)
    baseline_centers = windows.center_indices.astype(int)
    shared = np.intersect1d(neuro_centers, baseline_centers)
    if len(shared) != {"discrete": 16, "continuous": 4096}[model]:
        raise ValueError(f"Unexpected common-center count: {len(shared)}")
    methods = [(neuro_centers, neuro), (baseline_centers, DTWShift(windows).shifts),
               (baseline_centers, CTWShift(windows).shifts), (baseline_centers, FFTShift(windows, is_filter=False).shifts)]
    common_ms = []
    print("method     native N   native RMSE (ms)   shared N   shared RMSE (ms)")
    for index, (centers, estimated) in enumerate(methods):
        squared_error = (estimated - truth[centers]) ** 2
        native = np.sqrt(squared_error.mean()) * 1000 / FS
        shared_error = np.sqrt(squared_error[np.isin(centers, shared)].mean()) * 1000 / FS
        np.testing.assert_allclose(native, archived[index], rtol=0, atol=0.0005)
        common_ms.append(shared_error)
        print(f"{METHODS[index]:8}   {len(centers):6d}       {native:10.6f}       {len(shared):4d}         {shared_error:10.6f}")
    published = {"discrete": [0.069, 1.483, 2.257, 24.129], "continuous": [0.293, 0.724, 1.317, 11.331]}
    np.testing.assert_allclose(common_ms, published[model], rtol=0, atol=0.0005)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path, help="Path to the NeuroTD revision checkout")
    repo = parser.parse_args().repository.resolve()
    sys.path.insert(0, str(repo / "src"))
    for model in ("discrete", "continuous"):
        errors, scores = read_grid(repo, model)
        check_rankings(model, errors, scores)
        check_fixed_example(repo, model, errors[1, WINDOWS.index(W)])
    print("Both fixed examples and all 132 grid rows verified.")


if __name__ == "__main__":
    main()
