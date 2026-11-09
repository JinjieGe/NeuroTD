"""Checkpointed process pool. Workers do not share result files."""

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

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone

import numpy as np

from science import (
    CONFIG_HASH,
    EXPERIMENT_DIR,
    SPLITS,
    WINDOWS,
    atomic_savez,
    checkpoint_path,
    condition_by_id,
    load_checkpoint,
    load_clean_inputs,
    make_rng,
    observe_pair,
    observed_pair,
    pack_meta,
    pair_for_record,
    result_to_arrays,
    science_token,
    stored_result_is_consistent,
)

INPUT_PATH = os.path.join(EXPERIMENT_DIR, "inputs", "clean_signals.npz")
_CLEAN = None
_SOURCE_HASH = None


def init_worker(source_hash: str):
    global _CLEAN, _SOURCE_HASH
    _CLEAN = load_clean_inputs(INPUT_PATH)
    _SOURCE_HASH = source_hash


def _result_meta(cond, split, rep, pair_hash, result, elapsed):
    selected = int(result["selected_index"])
    return {
        "condition_id": int(cond),
        "split_id": int(split),
        "replicate_id": int(rep),
        "seed_entropy": [202610190243, int(cond), int(split), int(rep)],
        "input_hash": pair_hash,
        "config_hash": CONFIG_HASH,
        "source_hash": _SOURCE_HASH,
        "wrapper_revision": 1,
        "selected_index": selected,
        "selection_tie": bool(result["selection_tie"]),
        "selected_W": int(WINDOWS[selected]) if selected >= 0 else -1,
        "elapsed_sec": float(elapsed),
        "status": "ok",
        "pid": os.getpid(),
    }


def execute_formal(task):
    cond, split, rep = (int(task[0]), int(task[1]), int(task[2]))
    path = checkpoint_path(cond, split, rep)
    try:
        if os.path.exists(path):
            try:
                meta, arrays = load_checkpoint(path)
                ok, reason = stored_result_is_consistent(meta, arrays, _CLEAN, _SOURCE_HASH)
            except Exception:
                ok, reason = False, "unreadable"
                meta, arrays = None, None
            if ok:
                arrays = {k: v for k, v in arrays.items() if k != "meta_json"}
                return {
                    "ok": True,
                    "skipped": True,
                    "task": [cond, split, rep],
                    "path": path,
                    "elapsed_sec": 0.0,
                    "token": science_token(meta, arrays),
                    "model": condition_by_id(cond)["model"],
                }
            quarantine = path + ".invalid"
            if os.path.exists(quarantine):
                quarantine = path + f".invalid.{time.time_ns()}"
            os.replace(path, quarantine)
        t0 = time.perf_counter()
        x_obs, y_obs, pair_hash = pair_for_record(_CLEAN, cond, split, rep)
        spec = condition_by_id(cond)
        kind = SPLITS[split]["signal"]
        result = observe_pair(x_obs, y_obs, _CLEAN, spec["model"], kind)
        elapsed = time.perf_counter() - t0
        meta = _result_meta(cond, split, rep, pair_hash, result, elapsed)
        arrays = result_to_arrays(result)
        token = science_token(meta, arrays)
        arrays["meta_json"] = pack_meta(meta)
        atomic_savez(path, arrays)
        return {
            "ok": True,
            "skipped": False,
            "task": [cond, split, rep],
            "path": path,
            "elapsed_sec": elapsed,
            "token": token,
            "model": spec["model"],
            "selected_W": meta["selected_W"],
        }
    except Exception:
        return {
            "ok": False,
            "task": [cond, split, rep],
            "error": traceback.format_exc(),
        }


def execute_pilot(task):
    """Timing-only record. split_id 99 is outside the formal design."""
    cond, rep = (int(task[0]), int(task[1]))
    spec = condition_by_id(cond)
    try:
        t0 = time.perf_counter()
        rng = make_rng(cond, 99, rep)
        x_obs, y_obs = observed_pair(_CLEAN, spec, "dynamic", rng)
        observe_pair(x_obs, y_obs, _CLEAN, spec["model"], "dynamic")
        elapsed = time.perf_counter() - t0
        return {"ok": True, "task": [cond, rep], "elapsed_sec": elapsed, "model": spec["model"], "pid": os.getpid()}
    except Exception:
        return {"ok": False, "task": [cond, rep], "error": traceback.format_exc()}


def audit_task(task):
    """Recompute one formal record with the original functions and compare bitwise."""
    cond, split, rep = (int(task[0]), int(task[1]), int(task[2]))
    path = checkpoint_path(cond, split, rep)
    meta, arrays = load_checkpoint(path)
    ok, reason = stored_result_is_consistent(meta, arrays, _CLEAN, _SOURCE_HASH)
    x_obs, y_obs, pair_hash = pair_for_record(_CLEAN, cond, split, rep)
    spec = condition_by_id(cond)
    result = observe_pair(x_obs, y_obs, _CLEAN, spec["model"], SPLITS[split]["signal"])
    fresh = result_to_arrays(result)
    mismatched = []
    for key, value in fresh.items():
        if key not in arrays or not np.array_equal(np.asarray(value), np.asarray(arrays[key])):
            mismatched.append(key)
    return {
        "task": [cond, split, rep],
        "stored_consistent": ok,
        "stored_reason": reason,
        "input_hash_ok": meta.get("input_hash") == pair_hash,
        "mismatched": mismatched,
        "passed": ok and not mismatched and meta.get("input_hash") == pair_hash,
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp, path)


class RunLog:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.handle = open(path, "a", encoding="utf-8")

    def line(self, text):
        message = f"{utc_now()} {text}"
        print(message, flush=True)
        self.handle.write(message + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()


def run_pool(tasks, worker_fn, n_workers, source_hash, log, progress_path, phase_name, shrink_state):
    """Run tasks on a spawn pool. shrink_state['stop'] asks for an orderly reduction."""
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    pending = list(tasks)
    failures = []
    done = 0
    skipped = 0
    attempted = {}
    t_start = time.perf_counter()
    recent = []

    def record_progress():
        elapsed = time.perf_counter() - t_start
        rate = done / elapsed * 3600.0 if elapsed > 0 else 0.0
        cont_times = [item for item in recent if item > 5.0]
        eta = None
        if cont_times and pending:
            # remaining formal work is dominated by continuous records; caller updates a better ETA
            eta = None
        write_json(
            progress_path,
            {
                "phase": phase_name,
                "updated": utc_now(),
                "n_workers": n_workers,
                "n_done_this_call": done,
                "n_skipped_this_call": skipped,
                "n_pending_this_call": len(pending),
                "n_failed": len(failures),
                "wall_sec": elapsed,
                "completed_per_hour_this_call": rate,
            },
        )

    pool_breaks = 0
    while pending:
        pool_broke = False
        try:
            with ProcessPoolExecutor(
                max_workers=n_workers,
                mp_context=ctx,
                initializer=init_worker,
                initargs=(source_hash,),
                max_tasks_per_child=16,
            ) as executor:
                inflight = {}

                def submit_next():
                    if shrink_state.get("requested") or not pending:
                        return False
                    task = pending.pop(0)
                    fut = executor.submit(worker_fn, task)
                    inflight[fut] = task
                    return True

                for _ in range(n_workers):
                    if not submit_next():
                        break
                while inflight:
                    finished, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED, timeout=60)
                    if not finished:
                        record_progress()
                        log.line(
                            f"{phase_name} heartbeat workers={n_workers} inflight={len(inflight)} "
                            f"pending={len(pending)} done={done}"
                        )
                        continue
                    for fut in finished:
                        task = inflight.pop(fut)
                        try:
                            payload = fut.result()
                        except Exception as exc:
                            payload = {"ok": False, "task": list(task), "error": traceback.format_exc()}
                            if type(exc).__name__ == "BrokenProcessPool":
                                pool_broke = True
                                pending.insert(0, tuple(task))
                                for other_task in list(inflight.values()):
                                    pending.insert(0, tuple(other_task))
                                inflight.clear()
                                log.line(f"process pool broke; {len(pending)} tasks will be requeued")
                                break
                        if pool_broke:
                            break
                        if not payload.get("ok"):
                            key = tuple(payload.get("task", task))
                            attempted[key] = attempted.get(key, 0) + 1
                            if attempted[key] <= 1:
                                pending.append(tuple(key))
                                log.line(f"retry {key}")
                            else:
                                failures.append(payload)
                                fail_path = os.path.join(
                                    EXPERIMENT_DIR,
                                    "failures",
                                    f"c{key[0]}_s{key[1] if len(key) > 2 else 99}_r{key[-1]:04d}.json",
                                )
                                write_json(fail_path, payload)
                                log.line(f"failed {key}")
                        else:
                            done += 1
                            if payload.get("skipped"):
                                skipped += 1
                            elif payload.get("elapsed_sec"):
                                recent.append(float(payload["elapsed_sec"]))
                                if len(recent) > 40:
                                    recent = recent[-40:]
                        if not shrink_state.get("requested"):
                            try:
                                submit_next()
                            except Exception:
                                pool_broke = True
                                log.line("submit failed; recreating the pool")
                                break
                    if pool_broke:
                        break
                    if done % 5 == 0:
                        record_progress()
        except Exception as exc:
            if type(exc).__name__ != "BrokenProcessPool":
                raise
            pool_broke = True
            log.line(f"pool shutdown raised {type(exc).__name__}")
        if pool_broke:
            pool_breaks += 1
            if pool_breaks > 5:
                log.line("stopping after repeated pool failures")
                break
            continue
        pool_breaks = 0
        if shrink_state.get("requested") and pending:
            new_n = max(1, n_workers - 1)
            log.line(f"reducing workers {n_workers} -> {new_n} because {shrink_state.get('reason')}")
            n_workers = new_n
            shrink_state["requested"] = False
            shrink_state["n_workers"] = n_workers
            continue
        break
    record_progress()
    return {"done": done, "skipped": skipped, "failures": failures, "n_workers": n_workers, "recent_sec": recent}


def run_consistency(source_hash, log) -> dict:
    """Serial versus two-worker results, then a resume of the same formal tasks."""
    import multiprocessing as mp

    tasks = [(0, 0, 0), (3, 0, 0), (6, 4, 0)]
    # Remove any partial leftovers so both modes compute.
    saved = []
    for task in tasks:
        path = checkpoint_path(*task)
        if os.path.exists(path):
            os.replace(path, path + ".pre_consistency")
    ctx = mp.get_context("spawn")
    serial_tokens = {}
    with ProcessPoolExecutor(
        max_workers=1, mp_context=ctx, initializer=init_worker, initargs=(source_hash,)
    ) as executor:
        for task in tasks:
            payload = executor.submit(execute_formal, task).result()
            if not payload.get("ok") or payload.get("skipped"):
                raise RuntimeError(f"serial consistency task failed: {payload}")
            serial_tokens[tuple(task)] = payload["token"]
            saved.append(payload)
    parallel_tokens = {}
    # Compute the same tasks into a side directory by temporarily moving formal files.
    moved = []
    for task in tasks:
        path = checkpoint_path(*task)
        side = path + ".serial"
        os.replace(path, side)
        moved.append((path, side))
    with ProcessPoolExecutor(
        max_workers=2, mp_context=ctx, initializer=init_worker, initargs=(source_hash,)
    ) as executor:
        futures = [executor.submit(execute_formal, task) for task in tasks]
        for fut in futures:
            payload = fut.result()
            if not payload.get("ok") or payload.get("skipped"):
                raise RuntimeError(f"parallel consistency task failed: {payload}")
            parallel_tokens[tuple(payload["task"])] = payload["token"]
    mismatches = []
    for task, token in serial_tokens.items():
        if parallel_tokens.get(tuple(task)) != token:
            mismatches.append([list(task), token, parallel_tokens.get(tuple(task))])
    # Resume must skip the parallel outputs and keep the same token.
    resume_tokens = {}
    with ProcessPoolExecutor(
        max_workers=2, mp_context=ctx, initializer=init_worker, initargs=(source_hash,)
    ) as executor:
        for task in tasks:
            payload = executor.submit(execute_formal, task).result()
            if not payload.get("ok") or not payload.get("skipped"):
                raise RuntimeError(f"resume did not skip a valid checkpoint: {payload}")
            resume_tokens[tuple(payload["task"])] = payload["token"]
    # A truncated file must not count as success.
    bad = checkpoint_path(*tasks[0])
    with open(bad, "wb") as handle:
        handle.write(b"not a checkpoint")
    with ProcessPoolExecutor(
        max_workers=1, mp_context=ctx, initializer=init_worker, initargs=(source_hash,)
    ) as executor:
        payload = executor.submit(execute_formal, tasks[0]).result()
    recomputed = payload.get("ok") and not payload.get("skipped") and payload.get("token") == serial_tokens[tasks[0]]
    report = {
        "tasks": [list(task) for task in tasks],
        "serial_equals_parallel": not mismatches,
        "mismatches": mismatches,
        "resume_skipped": resume_tokens == parallel_tokens,
        "corrupt_file_recomputed": bool(recomputed),
        "elapsed_sec": {str(list(task)): item["elapsed_sec"] for item, task in zip(saved, tasks)},
    }
    if mismatches or not report["resume_skipped"] or not recomputed:
        raise RuntimeError(f"consistency failed: {json.dumps(report)}")
    log.line(
        "consistency passed: serial, two workers, resume skip, and corrupt-file recompute agree bitwise"
    )
    return report
