"""Run the paper scripts with additive ground-truth validation, in fresh folders."""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = {
    "discrete": "fig2_simulation_discrete_shift.py",
    "continuous": "fig3_simulation_continuous_shift.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--simulation", choices=["discrete", "continuous", "both"], default="discrete")
    parser.add_argument("--smoke", action="store_true", help="Only noise 0/0.02 and W=65/513; not a full benchmark")
    parser.add_argument("--seeds", nargs="+", type=int, help="Repeat per seed; default paper seeds: 0 / 2025")
    parser.add_argument("--output-dir", type=Path, help="NEW directory, must not exist")
    args = parser.parse_args()
    if args.seeds and (len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0):
        parser.error("Seeds must be distinct nonnegative integers")
    output = (args.output_dir or ROOT / "validation_runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    if output.exists():
        parser.error(f"Output directory already exists: {output}")
    output.mkdir(parents=True)
    simulations = list(SCRIPTS) if args.simulation == "both" else [args.simulation]
    tracked_sources = sorted((ROOT / "src" / "neurotd").glob("*.py")) + [Path(__file__).resolve()]
    metadata = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "smoke": args.smoke,
        "sampling_rate_hz": 5120, "runs": [], "dependencies": {},
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in tracked_sources},
    }
    for package in ("numpy", "scipy", "pandas", "matplotlib", "tslearn", "dtw-python", "scikit-image"):
        try:
            metadata["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            metadata["dependencies"][package] = "not installed"
    for label, command in (("git_commit", ["rev-parse", "HEAD"]), ("git_status", ["status", "--short"])):
        try:
            metadata[label] = subprocess.check_output(["git", "-C", str(ROOT), *command],
                                                       text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            metadata[label] = "unavailable (e.g. ZIP download)"
    manifest = output / "run_manifest.json"
    for simulation in simulations:
        seeds = args.seeds if args.seeds is not None else [0 if simulation == "discrete" else 2025]
        for seed in seeds:
            folder = output / f"{simulation}_seed_{seed}"
            folder.mkdir()
            env = os.environ.copy()
            env.update({"PYTHONPATH": str(ROOT / "src"), "MPLBACKEND": "Agg",
                        "MPLCONFIGDIR": str(output / ".matplotlib"),
                        "XDG_CACHE_HOME": str(output / ".cache"),
                        "NEUROTD_SIM_SEED": str(seed), "NEUROTD_SIM_SMOKE": "1" if args.smoke else "0"})
            command = [sys.executable, "-u", str(ROOT / "src" / "neurotd" / SCRIPTS[simulation])]
            run = {"simulation": simulation, "seed": seed, "directory": str(folder),
                   "command": command, "status": "running"}
            metadata["runs"].append(run)
            manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Running {simulation}, seed={seed}. Log: {folder / 'run.log'}", flush=True)
            started = time.monotonic()
            with (folder / "run.log").open("w") as log:
                result = subprocess.run(command, cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT)
            run.update(status="completed" if result.returncode == 0 else "failed", exit_code=result.returncode,
                       elapsed_seconds=time.monotonic() - started,
                       finished_utc=datetime.now(timezone.utc).isoformat())
            manifest.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            if result.returncode:
                raise SystemExit(f"Simulation failed; inspect {folder / 'run.log'}. Exit {result.returncode}")
    # Concatenate descriptive audits; do not pool overlapping windows as replicates.
    import pandas as pd
    files = sorted(output.glob("*_seed_*/known_delay_validation_*/window_selection.csv"))
    pd.concat([pd.read_csv(p) for p in files], ignore_index=True).to_csv(output / "all_window_selections.csv", index=False)
    print(f"Complete. Validation and source/environment manifest: {output}", flush=True)


if __name__ == "__main__":
    main()
