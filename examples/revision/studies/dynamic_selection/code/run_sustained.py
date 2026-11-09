"""Sustained pilot at 4 and 5 workers, with CPU speed-limit samples."""

import json
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import (
    ProcessPoolExecutor_from,
    cpu_speed_limit,
    memory_free_fraction,
    np_median,
    swap_used_mb,
)
from runner import execute_pilot
from science import EXPERIMENT_DIR, load_clean_inputs, scientific_source_hash, source_file_hashes
import multiprocessing as mp


def run_level(ctx, n_workers, source_hash, replicate_base):
    tasks = [(3 + (i % 4), replicate_base + i) for i in range(max(4, 2 * n_workers))]
    samples = []
    stop = threading.Event()

    def monitor():
        while not stop.wait(2.0):
            samples.append(
                {
                    "speed_limit": cpu_speed_limit(),
                    "free": memory_free_fraction(),
                    "swap_mb": swap_used_mb()[0],
                }
            )

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.perf_counter()
    completions = []
    with ProcessPoolExecutor_from(ctx, n_workers, source_hash) as executor:
        from concurrent.futures import as_completed

        futures = [executor.submit(execute_pilot, task) for task in tasks]
        for fut in as_completed(futures):
            payload = fut.result()
            if not payload.get("ok"):
                raise RuntimeError(payload)
            completions.append((time.perf_counter(), float(payload["elapsed_sec"])))
    stop.set()
    thread.join(timeout=3)
    completions.sort()
    wall = time.perf_counter() - started
    if len(completions) > n_workers:
        steady = (len(completions) - n_workers) / max(
            completions[-1][0] - completions[n_workers - 1][0], 1e-9
        ) * 3600.0
    else:
        steady = len(completions) / wall * 3600.0
    speeds = [s["speed_limit"] for s in samples if s["speed_limit"] is not None]
    return {
        "n_workers": n_workers,
        "n_tasks": len(tasks),
        "wall_sec": wall,
        "steady_records_per_hour": steady,
        "median_compute_sec": float(np_median([item[1] for item in completions])),
        "min_speed_limit": min(speeds) if speeds else None,
        "max_speed_limit": max(speeds) if speeds else None,
        "samples": samples,
    }


def main():
    clean = load_clean_inputs(os.path.join(EXPERIMENT_DIR, "inputs", "clean_signals.npz"))
    source_hash = scientific_source_hash(clean["continuous_source_sha256"], source_file_hashes())
    ctx = mp.get_context("spawn")
    results = []
    for n_workers, base in ((5, 2000), (4, 3000)):
        print(f"sustained n={n_workers}", flush=True)
        results.append(run_level(ctx, n_workers, source_hash, base))
        print(json.dumps({k: v for k, v in results[-1].items() if k != "samples"}), flush=True)
    path = os.path.join(EXPERIMENT_DIR, "pilot", "sustained_extra.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
        handle.write("\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
