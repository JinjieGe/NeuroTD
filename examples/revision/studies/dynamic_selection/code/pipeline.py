"""Run the frozen 2.4.3 simulation outside the original project tree."""

from __future__ import annotations
from revision_runtime import resolve_path

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
os.environ.setdefault("MPLCONFIGDIR", resolve_path('dynamic_selection', 'mplconfig'))

from pathlib import Path
import fcntl
import json
import math
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from science import (  # noqa: E402
    CONFIG_HASH,
    EXPERIMENT_DIR,
    PROJECT_DIR,
    expected_task_count,
    iter_tasks,
    load_clean_inputs,
    scientific_source_hash,
    source_file_hashes,
)
from runner import (  # noqa: E402
    INPUT_PATH,
    RunLog,
    execute_pilot,
    init_worker,
    run_consistency,
    run_pool,
    utc_now,
    write_json,
)


LOCK_PATH = os.path.join(EXPERIMENT_DIR, "run.lock")


def _sysctl(name):
    try:
        return subprocess.check_output(["sysctl", "-n", name], text=True).strip()
    except Exception as exc:
        return f"unknown:{type(exc).__name__}"


def _read(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception as exc:
        return f"unknown:{type(exc).__name__}:{exc}"


def memory_free_fraction():
    text = _read(["memory_pressure"])
    for line in text.splitlines():
        if "System-wide memory free percentage" in line:
            try:
                return float(line.split(":")[1].strip().rstrip("%")) / 100.0
            except ValueError:
                return None
    return None


def cpu_speed_limit():
    """100 means the scheduler is not limiting CPU speed. Lower values indicate throttling."""
    text = _read(["pmset", "-g", "therm"])
    for line in text.splitlines():
        if "CPU_Speed_Limit" in line and "=" in line:
            try:
                return int(line.split("=")[1].strip())
            except ValueError:
                return None
    return None


def swap_used_mb():
    text = _sysctl("vm.swapusage")
    # total = 3072.00M  used = 2249.75M  free = 822.25M
    for part in text.split():
        if part.endswith("M") and part[:-1].replace(".", "", 1).isdigit():
            pass
    used = None
    if "used =" in text:
        try:
            used = float(text.split("used =")[1].split("M")[0].strip())
        except ValueError:
            used = None
    return used, text


def take_snapshot(source_hash, file_hashes):
    free_frac = memory_free_fraction()
    swap_mb, swap_text = swap_used_mb()
    total = int(_sysctl("hw.memsize")) if _sysctl("hw.memsize").isdigit() else None
    snap = {
        "time": utc_now(),
        "hostname": _read(["hostname"]).strip(),
        "os": _read(["uname", "-a"]).strip(),
        "macos": _read(["sw_vers"]).strip(),
        "cpu_brand": _sysctl("machdep.cpu.brand_string"),
        "physical_cores": _sysctl("hw.physicalcpu"),
        "logical_cpus": _sysctl("hw.logicalcpu"),
        "os_cpu_count": os.cpu_count(),
        "sched_affinity": "unavailable: os.sched_getaffinity is absent on this macOS Python",
        "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "slurm_cpus": os.environ.get("SLURM_CPUS_PER_TASK"),
        "pbs": os.environ.get("PBS_JOBID"),
        "nslots": os.environ.get("NSLOTS"),
        "cpu_available_pmset": "see therm text",
        "loadavg": list(os.getloadavg()),
        "thermal_level": _sysctl("machdep.xcpm.cpu_thermal_level"),
        "pmset_therm": _read(["pmset", "-g", "therm"]),
        "memory_bytes": total,
        "memory_free_fraction": free_frac,
        "swap": swap_text.strip(),
        "swap_used_mb": swap_mb,
        "disk": _read(["df", "-h", str(Path.home()), EXPERIMENT_DIR]),
        "gpu": "Radeon Pro 580X, 8 GB, recorded only, not used",
        "python": sys.version,
        "executable": sys.executable,
        "config_hash": CONFIG_HASH,
        "source_hash": source_hash,
        "git": "not a git repository; commit object unavailable",
        "time_varying_shift_note": (
            "Current src/neurotd/time_varying_shift.py SHA-256 is "
            "09493e0070feb66251ca212ae05cf556013fef15b9b5d58bf19a616beefb7e84. "
            "Reversing _repro_diff_vs_1a6c4f2/time_varying_shift.py.diff in a temporary copy "
            "reproduces 28bb49eba3df7ea1e3d45d7afaea26a392c958ffae4aacbcc56b6b9cef68ce0b. "
            "The only added lines are optional progress callbacks. Calls omit progress, so those lines do not run. "
            "The project files were not modified."
        ),
    }
    write_json(os.path.join(EXPERIMENT_DIR, "environment", "system_snapshot.json"), snap)
    hash_path = os.path.join(EXPERIMENT_DIR, "environment", "python_source_sha256.txt")
    with open(hash_path, "w", encoding="utf-8") as handle:
        for rel, digest in file_hashes.items():
            handle.write(f"{digest}  {rel}\n")
    note = os.path.join(EXPERIMENT_DIR, "reports", "source_difference.md")
    os.makedirs(os.path.dirname(note), exist_ok=True)
    with open(note, "w", encoding="utf-8") as handle:
        handle.write(
            "# 源码差异\n\n"
            "本机工作区没有 `.git`，不能从提交对象复核 `1a6c4f23818bcfe3152f947c54c553a92edeca0c`。\n\n"
            "当前 `src/neurotd/time_varying_shift.py` 的 SHA-256 是 "
            "`09493e0070feb66251ca212ae05cf556013fef15b9b5d58bf19a616beefb7e84`。\n"
            "把 `_repro_diff_vs_1a6c4f2/time_varying_shift.py.diff` 逆向应用到临时副本后，"
            "SHA-256 等于指定的 `28bb49eba3df7ea1e3d45d7afaea26a392c958ffae4aacbcc56b6b9cef68ce0b`。\n"
            "差异只有可选的 progress 回调。本实验调用时不传入 progress，这些语句不会执行。\n"
            "估计器、E、信号模型和随机种子都没有改。原项目文件没有被覆盖或切换。\n\n"
            "`fig3_simulation_continuous_shift.py` 的已存差异从第 294 行附近的进度条开始，"
            "连续信号生成函数在这之前，不在该差异里。\n"
            "`neuro_signal.py` 和 `utils.py` 没有对应的差异文件；由于没有 git 对象，不能把它们单独认证为该提交。\n"
            "实验只读调用当前工作区里的原函数。\n"
        )
    return snap


def hard_cap_from_snapshot(snap) -> int:
    physical = int(snap["physical_cores"])
    reserve = max(1, math.ceil(0.2 * physical))
    cap = physical - reserve
    load1 = float(snap["loadavg"][0])
    occupied = int(round(load1))
    # Existing runnable work lowers the cap. Keep at least 2 so the pilot can compare.
    cap = min(cap, max(2, physical - occupied))
    return int(cap)


def _pilot_tasks(n_workers):
    n = max(4, 2 * n_workers)
    tasks = []
    for i in range(n):
        tasks.append((3 + (i % 4), i))  # conditions 3,4,5,6
    return tasks


def run_pilot(source_hash, snap, log) -> dict:
    import multiprocessing as mp

    cap = hard_cap_from_snapshot(snap)
    candidates = [1, 2, 4]
    if cap not in candidates and cap > 4:
        candidates.append(cap)
    candidates = [n for n in candidates if n <= cap]
    if not candidates:
        candidates = [1]
    free = snap.get("memory_free_fraction")
    total = snap.get("memory_bytes") or 0
    mem_budget = 0.7 * free * total if free is not None and total else None
    swap_start, _ = swap_used_mb()
    thermal0 = snap.get("thermal_level")
    log.line(
        f"pilot cap={cap} candidates={candidates} mem_budget_bytes={mem_budget} swap_start_mb={swap_start}"
    )
    ctx = mp.get_context("spawn")
    levels = []
    for n_workers in candidates:
        tasks = _pilot_tasks(n_workers)
        samples = []
        stop = threading.Event()

        def monitor():
            while not stop.wait(2.0):
                swap_mb, _text = swap_used_mb()
                samples.append(
                    {
                        "t": time.time(),
                        "load": list(os.getloadavg()),
                        "free": memory_free_fraction(),
                        "swap_mb": swap_mb,
                        "thermal": _sysctl("machdep.xcpm.cpu_thermal_level"),
                        "speed_limit": cpu_speed_limit(),
                        "rss_children": _child_rss(),
                    }
                )

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        level_swap_start, _level_swap_text = swap_used_mb()
        started = time.perf_counter()
        completions = []
        with ProcessPoolExecutor_from(ctx, n_workers, source_hash) as executor:
            futures = [executor.submit(execute_pilot, task) for task in tasks]
            from concurrent.futures import as_completed

            for fut in as_completed(futures):
                payload = fut.result()
                if not payload.get("ok"):
                    stop.set()
                    raise RuntimeError(payload)
                completions.append((time.perf_counter(), float(payload["elapsed_sec"])))
        stop.set()
        thread.join(timeout=3)
        wall = time.perf_counter() - started
        completions.sort()
        if len(completions) > n_workers:
            t_ready = completions[n_workers - 1][0]
            t_last = completions[-1][0]
            steady = (len(completions) - n_workers) / max(t_last - t_ready, 1e-9) * 3600.0
        else:
            steady = len(completions) / wall * 3600.0
        overall = len(completions) / wall * 3600.0
        peak_rss = max((s["rss_children"] for s in samples), default=0)
        swap_end, _ = swap_used_mb()
        swap_delta = None if level_swap_start is None or swap_end is None else swap_end - level_swap_start
        therm_vals = []
        speed_vals = []
        for s in samples:
            try:
                therm_vals.append(int(str(s["thermal"]).strip()))
            except ValueError:
                pass
            if s.get("speed_limit") is not None:
                speed_vals.append(s["speed_limit"])
        level = {
            "n_workers": n_workers,
            "n_tasks": len(tasks),
            "conditions": [3, 4, 5, 6],
            "split_id": 99,
            "wall_sec": wall,
            "steady_records_per_hour": steady,
            "overall_records_per_hour": overall,
            "compute_sec": [item[1] for item in completions],
            "median_compute_sec": float(np_median([item[1] for item in completions])),
            "peak_child_rss_bytes": peak_rss,
            "swap_delta_mb": swap_delta,
            "max_thermal": max(therm_vals) if therm_vals else None,
            "min_speed_limit": min(speed_vals) if speed_vals else None,
            "mean_free_fraction": _mean([s["free"] for s in samples if s["free"] is not None]),
            "startup_note": "steady rate starts after the first wave of n_workers completions",
        }
        levels.append(level)
        log.line(
            f"pilot n={n_workers} steady_per_hour={steady:.1f} median_sec={level['median_compute_sec']:.2f} "
            f"peak_rss={peak_rss} swap_delta_mb={swap_delta} thermal={level['max_thermal']}"
        )
        write_json(os.path.join(EXPERIMENT_DIR, "pilot", "levels_partial.json"), {"levels": levels})

    eligible = []
    for level in levels:
        reasons = []
        if mem_budget is not None and level["peak_child_rss_bytes"] > mem_budget:
            reasons.append("rss above memory budget")
        if level["swap_delta_mb"] is not None and level["swap_delta_mb"] > 256:
            reasons.append("swap increased by more than 256 MB")
        # machdep.xcpm.cpu_thermal_level is not a throttle flag on this iMac.
        # CPU_Speed_Limit below 90 means the scheduler is slowing the CPU.
        if level.get("min_speed_limit") is not None and level["min_speed_limit"] < 90:
            reasons.append(f"CPU speed limit fell to {level['min_speed_limit']}")
        level["exclude_reasons"] = reasons
        if not reasons:
            eligible.append(level)
    if not eligible:
        eligible = [min(levels, key=lambda item: item["n_workers"])]
    best = max(eligible, key=lambda item: item["steady_records_per_hour"])
    cutoff = 0.95 * best["steady_records_per_hour"]
    chosen = min(
        [item for item in eligible if item["steady_records_per_hour"] >= cutoff],
        key=lambda item: item["n_workers"],
    )
    # Sustained check on the chosen width, with a fresh replicate block.
    sustained = _sustained(ctx, chosen["n_workers"], source_hash, len(levels))
    ratio = sustained["steady_records_per_hour"] / chosen["steady_records_per_hour"]
    stepped_down = False
    if ratio < 0.90:
        smaller = [item for item in eligible if item["n_workers"] < chosen["n_workers"]]
        if smaller:
            chosen = max(smaller, key=lambda item: item["n_workers"])
            stepped_down = True
    decision = {
        "time": utc_now(),
        "cap": cap,
        "levels": levels,
        "chosen_workers": chosen["n_workers"],
        "chosen_steady_records_per_hour": chosen["steady_records_per_hour"],
        "best_steady_records_per_hour": best["steady_records_per_hour"],
        "rule": "smallest eligible worker count at or above 95 percent of the best eligible throughput",
        "sustained": sustained,
        "sustained_ratio": ratio,
        "stepped_down_after_sustained": stepped_down,
        "single_worker_median_sec": next(item["median_compute_sec"] for item in levels if item["n_workers"] == 1),
        "speedup_vs_single": chosen["steady_records_per_hour"]
        / next(item["steady_records_per_hour"] for item in levels if item["n_workers"] == 1),
        "mem_budget_bytes": mem_budget,
        "thermal_before": thermal0,
        "pilot_not_in_formal_results": True,
    }
    # Checkpoint storage estimate from one formal file if present, else from median record size later.
    decision["checkpoint_storage_note"] = (
        "About 0.3 MB per record uncompressed before measurement; "
        "29379 records would be under about 15 GB, against 1.5 TB free."
    )
    write_json(os.path.join(EXPERIMENT_DIR, "pilot", "decision.json"), decision)
    log.line(
        f"chosen workers={decision['chosen_workers']} speedup={decision['speedup_vs_single']:.2f} "
        f"sustained_ratio={ratio:.2f} stepdown={stepped_down}"
    )
    return decision


def ProcessPoolExecutor_from(ctx, n_workers, source_hash):
    from concurrent.futures import ProcessPoolExecutor

    return ProcessPoolExecutor(
        max_workers=n_workers,
        mp_context=ctx,
        initializer=init_worker,
        initargs=(source_hash,),
        max_tasks_per_child=8,
    )


def _sustained(ctx, n_workers, source_hash, level_count):
    tasks = [(3 + (i % 4), 1000 + level_count * 100 + i) for i in range(max(4, 2 * n_workers))]
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
    wall = time.perf_counter() - started
    completions.sort()
    if len(completions) > n_workers:
        steady = (len(completions) - n_workers) / max(
            completions[-1][0] - completions[n_workers - 1][0], 1e-9
        ) * 3600.0
    else:
        steady = len(completions) / wall * 3600.0
    return {
        "n_workers": n_workers,
        "n_tasks": len(tasks),
        "wall_sec": wall,
        "steady_records_per_hour": steady,
        "median_compute_sec": float(np_median([item[1] for item in completions])),
        "max_thermal": _sysctl("machdep.xcpm.cpu_thermal_level"),
        "min_speed_limit": cpu_speed_limit(),
        "swap_used_mb": swap_used_mb()[0],
        "free_fraction": memory_free_fraction(),
    }


def np_median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def _mean(values):
    values = list(values)
    if not values:
        return None
    return float(sum(values) / len(values))


def _child_rss():
    text = _read(["ps", "-axo", "pid,ppid,rss,command"])
    if text.startswith("unknown"):
        return 0
    total = 0
    me = os.getpid()
    for line in text.splitlines()[1:]:
        parts = line.split(None, 3)
        if len(parts) < 4:
            continue
        pid, ppid, rss, command = parts
        if int(ppid) == me or int(pid) == me:
            if "python" in command or int(pid) == me:
                total += int(rss) * 1024
    return total


def acquire_lock():
    os.makedirs(EXPERIMENT_DIR, exist_ok=True)
    handle = open(LOCK_PATH, "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.seek(0)
        existing = handle.read()
        raise SystemExit(f"experiment lock is already held: {existing}")
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "started": utc_now()}) + "\n")
    handle.flush()
    return handle


def remaining_tasks(splits):
    missing = []
    for task in iter_tasks(splits):
        from science import checkpoint_path

        if not os.path.exists(checkpoint_path(*task)):
            missing.append(task)
    return missing


def main():
    try:
        os.nice(5)
    except Exception:
        pass
    lock = acquire_lock()
    log = RunLog(os.path.join(EXPERIMENT_DIR, "logs", "run.log"))
    try:
        assert expected_task_count() == 29379
        assert expected_task_count([0, 3, 4]) == 15393
        assert expected_task_count([1, 2]) == 13986
        clean = load_clean_inputs(INPUT_PATH)
        file_hashes = source_file_hashes()
        source_hash = scientific_source_hash(clean["continuous_source_sha256"], file_hashes)
        snap = take_snapshot(source_hash, file_hashes)
        write_json(
            os.path.join(EXPERIMENT_DIR, "process.json"),
            {"pid": os.getpid(), "started": utc_now(), "status": "running", "source_hash": source_hash},
        )
        log.line(f"source_hash={source_hash} config_hash={CONFIG_HASH}")
        bench_path = os.path.join(EXPERIMENT_DIR, "benchmarks", "benchmark_report.json")
        with open(bench_path, encoding="utf-8") as handle:
            bench = json.load(handle)
        if not bench.get("passed") or bench.get("config_hash") != CONFIG_HASH:
            raise SystemExit("benchmarks are not passed for this config; formal run stopped")
        if bench["tolerance"]["floats"] != "bitwise" or bench["tolerance"]["atol"] != 0.0:
            raise SystemExit("tolerance was not frozen at bitwise zero")
        cons_path = os.path.join(EXPERIMENT_DIR, "benchmarks", "consistency.json")
        if not os.path.exists(cons_path):
            report = run_consistency(source_hash, log)
            write_json(cons_path, report)
        else:
            log.line("consistency report already present")
        pilot_path = os.path.join(EXPERIMENT_DIR, "pilot", "decision.json")
        if not os.path.exists(pilot_path):
            decision = run_pilot(source_hash, snap, log)
        else:
            with open(pilot_path, encoding="utf-8") as handle:
                decision = json.load(handle)
            log.line(f"reusing pilot decision workers={decision['chosen_workers']}")
        n_workers = int(decision["chosen_workers"])
        shrink_state = {"requested": False, "reason": "", "n_workers": n_workers}
        stop_monitor = threading.Event()
        swap_origin, _ = swap_used_mb()
        bad_hits = {"n": 0}

        def monitor():
            while not stop_monitor.wait(30):
                swap_mb, _text = swap_used_mb()
                speed_limit = cpu_speed_limit()
                free = memory_free_fraction()
                bad = False
                reason = []
                # Pilot accepted 2 workers because 4–5 workers held the CPU speed limit near 65.
                # Shrink only if the limit falls below that already-rejected operating region.
                if speed_limit is not None and speed_limit < 60:
                    bad = True
                    reason.append(f"speed_limit={speed_limit}")
                if swap_origin is not None and swap_mb is not None and swap_mb > swap_origin + 512:
                    bad = True
                    reason.append(f"swap_mb={swap_mb}")
                if free is not None and free < 0.10:
                    bad = True
                    reason.append(f"free={free}")
                if bad:
                    bad_hits["n"] += 1
                else:
                    bad_hits["n"] = 0
                if bad_hits["n"] >= 2 and not shrink_state["requested"] and shrink_state["n_workers"] > 1:
                    shrink_state["requested"] = True
                    shrink_state["reason"] = ",".join(reason)
                    log.line("resource pressure " + shrink_state["reason"])

        thread = threading.Thread(target=monitor, daemon=True)
        thread.start()
        from analyze import analyze

        for phase_name, splits in (("phase1", [0, 3, 4]), ("phase2", [1, 2])):
            missing = remaining_tasks(splits)
            log.line(f"{phase_name} missing {len(missing)} of {expected_task_count(splits)}")
            if missing:
                outcome = run_pool(
                    missing,
                    __import__("runner").execute_formal,
                    shrink_state["n_workers"],
                    source_hash,
                    log,
                    os.path.join(EXPERIMENT_DIR, "progress.json"),
                    phase_name,
                    shrink_state,
                )
                log.line(
                    f"{phase_name} pool done={outcome['done']} skipped={outcome['skipped']} "
                    f"failures={len(outcome['failures'])} workers={outcome['n_workers']}"
                )
                if outcome["failures"]:
                    log.line(f"{phase_name} stopped with failures")
                    break
            else:
                log.line(f"{phase_name} already complete")
            try:
                analyze("interim" if phase_name == "phase1" else "final", source_hash)
            except Exception:
                import traceback

                log.line("analyze error\n" + traceback.format_exc())
        # Confirm the project sources did not change.
        fresh = source_file_hashes()
        if fresh != file_hashes:
            log.line("WARNING project python hashes changed during the run")
        else:
            log.line("project python hashes unchanged")
        write_json(
            os.path.join(EXPERIMENT_DIR, "process.json"),
            {"pid": os.getpid(), "finished": utc_now(), "status": "finished", "source_hash": source_hash},
        )
    finally:
        stop = locals().get("stop_monitor")
        if stop is not None:
            stop.set()
        log.line("pipeline exit")
        log.close()
        lock.close()


if __name__ == "__main__":
    main()
