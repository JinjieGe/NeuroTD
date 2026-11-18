"""Terminal + log progress reporting for the review 2.4.2 pipeline.

Writes every line to stdout and run.log, structured events to progress.jsonl,
and per-stage wall time to timing_summary.csv. Completed counts only change
when the caller reports a finished, checked unit; the heartbeat thread never
advances them.
"""
from __future__ import annotations

import csv
import json
import resource
import sys
import threading
import time
from datetime import datetime
from pathlib import Path


def fmt_hms(seconds: float | None) -> str:
    if seconds is None or seconds != seconds:
        return "--:--:--"
    seconds = int(round(seconds))
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"


def peak_rss_mb() -> tuple[float, float]:
    """(self, children) peak RSS in MB. macOS reports bytes."""
    scale = 1 / (1024 * 1024) if sys.platform == "darwin" else 1 / 1024
    own = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    return own, kids


class _Tee:
    def __init__(self, stream, logfile):
        self.stream = stream
        self.logfile = logfile

    def write(self, s):
        self.stream.write(s)
        self.stream.flush()
        self.logfile.write(s)
        self.logfile.flush()

    def flush(self):
        self.stream.flush()
        self.logfile.flush()


class Reporter:
    def __init__(self, run_dir: Path, heartbeat_s: float = 10.0):
        self.run_dir = Path(run_dir)
        self.log_f = open(self.run_dir / "run.log", "a", buffering=1, encoding="utf-8")
        self.jsonl_f = open(self.run_dir / "progress.jsonl", "a", buffering=1, encoding="utf-8")
        self.timing_path = self.run_dir / "timing_summary.csv"
        self._lock = threading.RLock()
        sys.stderr = _Tee(sys.__stderr__, self.log_f)
        self.heartbeat_s = heartbeat_s
        self.t0 = time.time()
        self.stage = None
        self.state: dict = {}
        self._stop = threading.Event()
        self._hb = threading.Thread(target=self._heartbeat, daemon=True)
        self._hb.start()

    # ------------------------------------------------------------------ output
    def line(self, text: str = ""):
        with self._lock:
            stamp = datetime.now().strftime("%H:%M:%S")
            for part in str(text).split("\n"):
                out = f"[{stamp}] {part}"
                sys.__stdout__.write(out + "\n")
                sys.__stdout__.flush()
                self.log_f.write(out + "\n")

    def event(self, kind: str, **payload):
        rec = {"time": datetime.now().isoformat(timespec="seconds"),
               "elapsed_s": round(time.time() - self.t0, 3),
               "stage": self.stage, "kind": kind, **payload}
        with self._lock:
            self.jsonl_f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")

    # ------------------------------------------------------------------ stages
    def start_stage(self, name: str):
        self.stage = name
        self.state = {"stage": name, "t_start": time.time(), "task": None,
                      "task_start": None, "done": None, "total": None,
                      "unit": None, "extra": None}
        self.line("=" * 78)
        self.line(f"[{name}] Starting stage")
        self.event("stage_start")

    def end_stage(self, status: str = "complete", note: str = ""):
        el = time.time() - self.state["t_start"]
        own, kids = peak_rss_mb()
        self.line(f"[{self.stage}] Stage ended: {status} | elapsed {fmt_hms(el)} ({el:.1f} s) "
                  f"| peak memory: parent {own:.0f} MB / children {kids:.0f} MB" + (f" | {note}" if note else ""))
        self.event("stage_end", status=status, seconds=round(el, 3), note=note,
                   peak_rss_self_mb=round(own, 1), peak_rss_children_mb=round(kids, 1))
        new = not self.timing_path.exists()
        with open(self.timing_path, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["stage", "status", "start", "end", "seconds",
                            "peak_rss_self_mb", "peak_rss_children_mb", "note"])
            w.writerow([self.stage, status,
                        datetime.fromtimestamp(self.state["t_start"]).isoformat(timespec="seconds"),
                        datetime.now().isoformat(timespec="seconds"), f"{el:.1f}",
                        f"{own:.0f}", f"{kids:.0f}", note])
        self.stage = None
        return el

    def set_task(self, task: str | None):
        self.state["task"] = task
        self.state["task_start"] = time.time() if task else None

    def set_progress(self, done=None, total=None, unit=None, extra=None):
        if done is not None:
            self.state["done"] = done
        if total is not None:
            self.state["total"] = total
        if unit is not None:
            self.state["unit"] = unit
        if extra is not None:
            self.state["extra"] = extra

    def _heartbeat(self):
        while not self._stop.wait(self.heartbeat_s):
            st = self.state
            if not st or not self.stage:
                continue
            el = time.time() - st["t_start"]
            if st.get("done") is None or st.get("total") is None:
                prog = "Progress unavailable"
            else:
                prog = f"Completed {st['done']}/{st['total']} {st.get('unit') or ''}".rstrip()
            task = st.get("task")
            tdur = ""
            if task and st.get("task_start"):
                tdur = f" (running for {fmt_hms(time.time() - st['task_start'])})"
            extra = f" | {st['extra']}" if st.get("extra") else ""
            self.line(f"[Heartbeat][{self.stage}] Stage elapsed {fmt_hms(el)} | {prog} | "
                      f"current task: {task or 'none'}{tdur}{extra}")

    def close(self):
        self._stop.set()
        self.log_f.flush()
        self.jsonl_f.flush()


class RateTracker:
    """Rate and ETA from real completed units only."""

    def __init__(self, min_units: int = 3, min_seconds: float = 5.0):
        self.t0 = time.time()
        self.min_units = min_units
        self.min_seconds = min_seconds

    def text(self, done: int, total: int, unit_label: str) -> str:
        el = time.time() - self.t0
        if done < self.min_units or el < self.min_seconds or done == 0:
            return f"Elapsed {fmt_hms(el)} | estimating speed and time remaining"
        per = el / done
        rem = per * max(total - done, 0)
        return f"Elapsed {fmt_hms(el)} | mean {per:.2f} s/{unit_label} | estimated time remaining {fmt_hms(rem)}"
