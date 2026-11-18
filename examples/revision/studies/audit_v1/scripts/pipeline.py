"""Review 2.4.2 re-test of Figure 4c (sub-01, ses-01..03): full pipeline with live progress.

Run from the repository root:
    python -u reanalysis_review2_4_2_20261031_1036/scripts/pipeline.py [--stages s1,s2,...]
"""
from __future__ import annotations
from revision_runtime import resolve_path

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(resolve_path('repo', ''))
OUT = Path(resolve_path("audit_v1"))
os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(OUT / "scripts"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import stats_core as sc  # noqa: E402
from progress import RateTracker, Reporter, fmt_hms, peak_rss_mb  # noqa: E402

DATA = Path(resolve_path("raw", "sub-01"))
NWB = {s: DATA / f"sub-01_ses-0{s}.nwb" for s in (1, 2, 3)}
OLD = ROOT / ("reproduction_audit_20261004/reviewer2_major5_data_reporting_20261004_1304/"
              "reconstruction_ieeg150_patchseq_istd_20261004_1413")
OLD_CKPT = OLD / "checkpoints/ieeg"
OLD_SHA = ROOT / "reproduction_audit_20261004/reviewer2_major5_data_reporting_20261004_1304/tables/ieeg_nwb_sha256.csv"
OLD_SRC_HASH = {
    "src/neurotd/app_iEEG/fig4_app_iEEG.py": "687f841a9674561c5bca0d55a0853baefced4b4371336843c337c4a824ffa719",
    "src/neurotd/hiEEG_functions.py": "4f55d4bc1b758543ba404579475d9b263ec3dcfdd016f707f113b250f067075e",
    "src/neurotd/time_varying_shift.py": "09493e0070feb66251ca212ae05cf556013fef15b9b5d58bf19a616beefb7e84",
    "src/neurotd/utils.py": "633946f30cc2f3683999a66c85b99d068a5b444a08c353dcc336c4e9d194e522",
}
TAB = OUT / "tables"
FIG = OUT / "figures"
CK = OUT / "checkpoints"
SYN = OUT / "synthetic_validation"

W, H, FS = 1001, 500, 32000
MAX_SHIFT = W // 2
TRIAL_LEN = 256000
MS_PER_SAMPLE = 1000.0 / FS
NEAR_ZERO = 1e-8
N_WORKERS = 6
SEED = 20261031
B_MAIN = 50000
B_REFINE = 500000
BATCH = 5000
LS = [1, 2, 5, 10]
REF_L = 5
SCOPES = {  # name -> (scope_code, sessions)
    "pooled_3_sessions": (0, [1, 2, 3]),
    "ses-01_only": (1, [1]), "ses-02_only": (2, [2]), "ses-03_only": (3, [3]),
    "drop_ses-01": (11, [2, 3]), "drop_ses-02": (12, [1, 3]), "drop_ses-03": (13, [1, 2]),
}
VARIANTS_POOLED = ["main_pooled_nonzero_all_trials", "A_pooled_including_zero", "B_pooled_artifact_excluded",
                   "D_pair_trial_session_equal_weight", "E_trial_median_difference", "F_trend_condition_adjusted"]
VARIANTS_SUB = VARIANTS_POOLED[:5]
ESTIMAND = {
    "main_pooled_nonzero_all_trials": "pooled mean |delay| (sum S / sum N), nonzero windows, all 150 trials",
    "A_pooled_including_zero": "pooled mean |delay| including finite zero-delay windows",
    "B_pooled_artifact_excluded": "pooled mean |delay|, nonzero windows, NWB artifact trials given zero weight",
    "D_pair_trial_session_equal_weight": "mean over sessions of mean over trials of mean over channel pairs of pair-mean nonzero |delay|",
    "E_trial_median_difference": "mean over sessions of mean over trials of (trial-region median A - median B), nonzero windows",
    "F_trend_condition_adjusted": "OLS-adjusted trial-equal mean difference of trial-level pooled means (session intercepts averaged; within-session trial trend, set_size, correct)",
}
REG_TRANSLATION = {  # copied from hiEEG_functions.load_iEEG_micro
    "Amyg": "Amygdala", "FuG": "Fusiform Gyrus", "Hipp": "Hippocampus", "INS": "Insular Gyrus",
    "IPL": "Inferior Parietal Lobule", "ITG": "Inferior Temporal Gyrus", "MTG": "Middle Temporal Gyrus",
    "PhG": "Parahippocampal Gyrus", "STG": "Superior Temporal Gyrus", "pSTS": "posterior Superior Temporal Sulcus",
}
HIST_MEDIANS = {"Amygdala": 9.28125, "Hippocampus": 10.5, "Middle Temporal Gyrus": 9.375,
                "Superior Temporal Gyrus": 8.875}
HIST_TOTAL_NONZERO = 4108812

STAGES = [
    ("s1", "Environment and input checks"), ("s2", "Verify and reuse earlier checkpoints"), ("s3", "Delay estimation"),
    ("s4", "Trial and region summaries"), ("s5", "Original Welch comparisons"), ("s6", "Bootstrap and sensitivity analysis"),
    ("s7", "Figures and report"), ("s8", "Output integrity checks"),
]
REP: Reporter | None = None
RUN_STATE_PATH = OUT / "run_state.json"


# ============================================================================ helpers
def load_state() -> dict:
    if RUN_STATE_PATH.exists():
        return json.loads(RUN_STATE_PATH.read_text())
    return {}


def save_state(**kw):
    st = load_state()
    st.update(kw)
    RUN_STATE_PATH.write_text(json.dumps(st, indent=2, ensure_ascii=False, default=str))


def sha256_file(path: Path, on_bytes=None, chunk=1 << 24) -> str:
    h = hashlib.sha256()
    done = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
            done += len(b)
            if on_bytes:
                on_bytes(done)
    return h.hexdigest()


def dec(v):
    return v.decode("utf-8", errors="replace") if isinstance(v, bytes) else v


def major_region(loc: str) -> str:
    p = loc.split(", ")[0]
    return REG_TRANSLATION.get(p, p)


def lateral_of(loc: str):
    parts = loc.split(", ")
    return parts[1].split(" ")[0] if len(parts) > 1 else None


def expected_centers():
    starts = np.arange(MAX_SHIFT, TRIAL_LEN - W - MAX_SHIFT + 1, H)
    return starts + (W - 1) // 2


# ============================================================================ worker
def _init_worker():
    import h5py  # noqa: F401
    from neurotd.time_varying_shift import compute_windowed_mse_with_hop  # noqa: F401


def _ping(_):
    return os.getpid()


def make_pool():
    # spawn, not fork: the parent runs a heartbeat thread
    pool = mp.get_context("spawn").Pool(N_WORKERS, initializer=_init_worker)
    pool.map(_ping, range(N_WORKERS * 4), chunksize=1)
    return pool


def _reestimate(task):
    """Run the original NeuroTD function on one trial x channel pair read from the NWB file."""
    import h5py
    from neurotd.time_varying_shift import compute_windowed_mse_with_hop
    ses, seg, k, ci, cj = task
    t0 = time.perf_counter()
    try:
        with h5py.File(NWB[ses], "r") as f:
            block = np.asarray(f["acquisition/ecephys.lfp/data"][seg * TRIAL_LEN:(seg + 1) * TRIAL_LEN, [ci, cj]],
                               dtype=np.float64)
        centers, shifts = compute_windowed_mse_with_hop(
            np.ascontiguousarray(block[:, 0]), np.ascontiguousarray(block[:, 1]), W, H,
            window_fun="rectangle", continuous=False)
        return dict(task=task, ok=True, shifts=np.asarray(shifts), centers=np.asarray(centers),
                    n_nonfinite=int((~np.isfinite(block)).sum()), seconds=time.perf_counter() - t0, error=None)
    except Exception:
        return dict(task=task, ok=False, shifts=None, centers=None, n_nonfinite=None,
                    seconds=time.perf_counter() - t0, error=traceback.format_exc())


def run_pair_tasks(tasks, label, on_done, expected_total=None, reused=0):
    """Run tasks in a worker pool; on_done(result) must return True if the result passed checks."""
    rt = RateTracker(min_units=N_WORKERS, min_seconds=10)
    n_ok = n_fail = 0
    windows = 0
    last_print = 0.0
    REP.set_progress(done=0, total=len(tasks), unit="pairs")
    REP.set_task(f"Starting {N_WORKERS} workers using the spawn method")
    with make_pool() as pool:
        REP.set_task(f"Submitted {len(tasks)} pairs; waiting for the first result")
        for res in pool.imap_unordered(_reestimate, tasks, chunksize=1):
            ses, seg, k, ci, cj = res["task"]
            passed = res["ok"] and on_done(res)
            if passed:
                n_ok += 1
                windows += int(res["shifts"].size)
            else:
                n_fail += 1
                REP.line(f"[{label}] Error or discrepancy: ses-0{ses}, trial {seg}, pair #{k} ({ci},{cj}). "
                         f"Recording failure and continuing this stage; failures will be reported at the end. Cause: {(res['error'] or 'check failed')[-400:]}")
            REP.set_progress(done=n_ok, total=len(tasks), unit="pairs",
                             extra=f"failed {n_fail}")
            REP.set_task(f"ses-0{ses} trial={seg} pair=({ci},{cj})")
            now = time.time()
            if now - last_print >= 10 or n_ok + n_fail == len(tasks):
                last_print = now
                pct = 100 * n_ok / max(len(tasks), 1)
                REP.line(f"[{label}]\n"
                         f"Session {ses}/3 | trial {seg + 1}/50 | last completed pair #{k} ({ci},{cj})\n"
                         f"Completed {n_ok}/{len(tasks)} pairs ({pct:.1f}%) | reused {reused} pairs | completed {windows} windows\n"
                         f"{rt.text(n_ok + n_fail, len(tasks), 'pair')}\n"
                         f"Failed/pending retry: {n_fail}")
            REP.event("pair_done", label=label, session=ses, trial=seg, pair=k, ci=ci, cj=cj,
                      ok=bool(passed), seconds=round(res["seconds"], 3), done=n_ok, failed=n_fail, total=len(tasks))
    REP.set_task(None)
    return n_ok, n_fail, windows


# ============================================================================ S1
def stage_s1():
    env = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
           "os_cpu_count": os.cpu_count(), "workers_used_for_delay_tasks": N_WORKERS}
    import importlib
    for m in ["numpy", "scipy", "pandas", "matplotlib", "seaborn", "h5py", "pynwb", "statsmodels", "tqdm"]:
        env[m] = importlib.import_module(m).__version__
    try:
        env["hardware"] = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True,
                                         text=True, timeout=30).stdout
    except Exception as e:  # noqa: BLE001
        env["hardware"] = f"unavailable: {e}"
    env["git"] = "not a git repository (no .git in workspace or parents); COMMIT UNKNOWN"
    REP.line(f"Python {sys.version.split()[0]} | numpy {env['numpy']} | scipy {env['scipy']} | pandas {env['pandas']} "
             f"| pynwb {env['pynwb']} | os.cpu_count={os.cpu_count()} | workers {N_WORKERS}")
    REP.line("Git: the original workspace had no Git metadata (commit unknown). Compare source hashes against NeuroTD-main.zip to identify changes.")

    # source hashes and comparison with the downloaded zip and the 2026-10-04 run
    rows = []
    zpath = ROOT.parent / "NeuroTD-main.zip"
    zmembers = {}
    if zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            for n in z.namelist():
                if n.endswith(".py") and "/src/neurotd/" in n:
                    rel = n.split("/", 1)[1]
                    zmembers[rel] = hashlib.sha256(z.read(n)).hexdigest()
    for p in sorted((ROOT / "src/neurotd").rglob("*.py")):
        rel = str(p.relative_to(ROOT))
        h = sha256_file(p)
        rows.append(dict(path=rel, sha256=h, bytes=p.stat().st_size,
                         mtime=datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                         sha256_in_NeuroTD_main_zip=zmembers.get(rel, "not in zip"),
                         same_as_zip=(zmembers.get(rel) == h) if rel in zmembers else None,
                         sha256_recorded_2026_10_04=OLD_SRC_HASH.get(rel, ""),
                         same_as_2026_10_04=(OLD_SRC_HASH.get(rel) == h) if rel in OLD_SRC_HASH else None))
    src = pd.DataFrame(rows)
    src.to_csv(TAB / "provenance_source_hashes.csv", index=False)
    changed = src[(src["same_as_zip"] == False)]["path"].tolist()  # noqa: E712
    key = src[src["path"].isin(OLD_SRC_HASH)]
    REP.line(f"Source: {len(src)} Python files; {len(changed)} differ from NeuroTD-main.zip: {changed[:6]}")
    REP.line("Hashes agree with the 2026-10-04 record: " + ", ".join(f"{Path(r.path).name}={'yes' if r.same_as_2026_10_04 else 'no'}"
                                               for r in key.itertuples()))

    # NWB hashes
    old = pd.read_csv(OLD_SHA)
    nwb_rows = []
    for s, p in NWB.items():
        size = p.stat().st_size
        REP.set_task(f"SHA-256 {p.name}")
        REP.set_progress(done=0, total=size, unit="bytes")
        t0 = time.time()
        h = sha256_file(p, on_bytes=lambda d: REP.set_progress(done=d))
        rec = old[old["path"].str.endswith(p.name)]
        prev = rec["sha256"].iloc[0] if len(rec) else ""
        nwb_rows.append(dict(session=f"ses-0{s}", path=str(p.relative_to(ROOT)), bytes=size, sha256=h,
                             sha256_recorded_2026_10_04=prev, match=(h == prev),
                             mtime=datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")))
        REP.line(f"NWB {p.name}: {size:,} bytes | SHA-256 {h[:16]}… | matches October 4: {h == prev} | {time.time() - t0:.1f} s")
        if h != prev:
            REP.line(f"Warning: {p.name} differs from its October 4 hash. Verify the earlier checkpoints before reuse.")
    REP.set_task(None)
    pd.DataFrame(nwb_rows).to_csv(TAB / "provenance_nwb_sha256.csv", index=False)

    # checkpoint manifest
    files = sorted(OLD_CKPT.glob("*"))
    REP.set_progress(done=0, total=len(files), unit="files")
    man = []
    for i, p in enumerate(files):
        man.append(dict(file=p.name, bytes=p.stat().st_size, sha256=sha256_file(p),
                        mtime=datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")))
        REP.set_progress(done=i + 1)
    pd.DataFrame(man).to_csv(TAB / "provenance_old_checkpoint_manifest.csv", index=False)
    REP.line(f"Earlier checkpoint directory {OLD_CKPT.relative_to(ROOT)}: recorded hashes for {len(files)} files.")
    REP.line("The original per-session files subject_1_session_{1,2,3}__neurotd_df_results.csv were unavailable in the historical search (see report).")

    # ---- representative benchmarks for the time estimate
    REP.line("Benchmark 1: original estimator, six channel pairs from ses-01 trial 0, six workers. Throughput is measured in parallel.")
    pairs = pd.read_csv(OLD_CKPT / "sub-01_ses-01_pairs.csv")
    bench_tasks = [(1, 0, k, int(pairs.channel_i[k]), int(pairs.channel_j[k])) for k in range(6)]
    REP.set_task("Benchmark: six parallel original-estimator evaluations")
    with make_pool() as pool:
        t0 = time.time()
        res = pool.map(_reestimate, bench_tasks, chunksize=1)
        par_wall = time.time() - t0
    z = np.load(OLD_CKPT / "sub-01_ses-01_trial00.npz")
    bench_match = all(r["ok"] and np.array_equal(r["shifts"], z["shifts_samples"][r["task"][2]].astype(int))
                      for r in res)
    per_pair_serial = float(np.mean([r["seconds"] for r in res]))
    per_pair_wall = par_wall / len(bench_tasks)
    REP.line(f"  Wall time for six pairs {par_wall:.1f} s; parallel throughput {per_pair_wall:.2f} s/pair, mean CPU time {per_pair_serial:.2f} s/pair; "
             f"all windows match earlier checkpoints: {bench_match}")

    REP.line("Benchmark 2: one batch of 5000 bootstrap draws on 150 synthetic trials and four regions, including regression variants.")
    rng = np.random.default_rng(0)
    T = 150
    S = rng.gamma(5, 2, (T, 4)) * 1000
    N = rng.integers(500, 1500, (T, 4)).astype(float)
    w = np.ones(T)
    slices = [(0, 50), (50, 100), (100, 150)]
    masks = [np.r_[np.zeros(a, bool), np.ones(b - a, bool), np.zeros(T - b, bool)] for a, b in slices]
    X = np.column_stack([m.astype(float) for m in masks] + [rng.normal(size=(T, 6))])
    t0 = time.time()
    idx = sc.stratified_indices(np.random.default_rng(1), slices, 5, BATCH)
    C = sc.counts_from_indices(idx, T)
    for _ in range(3):
        sc.pairwise_diffs(sc.pooled_means(C, S, N, w))
    sc.pairwise_diffs(sc.session_equal_means(C, S / N, w, masks))
    sc.session_equal_means(C, sc.pairwise_diffs(S / N), w, masks)
    sc.adjusted_regression(C, sc.pairwise_diffs(S / N), X, 3)
    t_batch_pooled = time.time() - t0
    REP.line(f"  One batch of 5000 draws (six variants): {t_batch_pooled:.2f} s")

    n_streams = len(SCOPES) * len(LS)
    est = {
        "1_read_and_check_s": "Measured S1 hashing plus about 30 s for S2 metadata",
        "2_delay_reestimation_s": round(96 * per_pair_wall, 1),
        "2b_full_recompute_if_needed_s": round(9000 * per_pair_wall, 1),
        "3_bootstrap_s": round(n_streams * (B_MAIN / BATCH) * t_batch_pooled, 1),
        "3b_bootstrap_refine_per_setting_s": round((B_REFINE / BATCH) * t_batch_pooled, 1),
        "4_figures_report_s": "About 120–300 s (rough estimate without a benchmark)",
        "per_pair_wall_s_6workers": per_pair_wall, "per_pair_cpu_s": per_pair_serial,
        "bootstrap_batch_s": t_batch_pooled,
    }
    (OUT / "logs/time_estimate_at_start.json").write_text(json.dumps(est, indent=2, ensure_ascii=False))
    REP.line("Initial runtime estimates:")
    REP.line(f"  1. Reading and checks: S1 measured; S2 metadata and checkpoint validation estimated at 30–60 s")
    REP.line(f"  2. Delay estimation: if all checkpoints pass, re-estimate 96 sampled pairs in about {fmt_hms(est['2_delay_reestimation_s'])}; "
             f"all 9000 pairs would take about {fmt_hms(est['2b_full_recompute_if_needed_s'])}")
    REP.line(f"  3. Resampling: {n_streams} index streams × {B_MAIN} draws, about {fmt_hms(est['3_bootstrap_s'])}; "
             f"if triggered, refining each configuration (B={B_REFINE}) adds about {fmt_hms(est['3b_bootstrap_refine_per_setting_s'])}")
    REP.line("  4. Figures and report: about 2–5 minutes (rough estimate)")
    env["time_estimate"] = est
    (OUT / "logs/environment.json").write_text(json.dumps(env, indent=2, ensure_ascii=False, default=str))
    save_state(benchmark_pairs_match=bench_match, per_pair_wall_s=per_pair_wall,
               nwb_hash_all_match=all(r["match"] for r in nwb_rows))
    # store benchmark results as verification checkpoints so they are not recomputed in S2
    vdir = CK / "reestimate"
    vdir.mkdir(exist_ok=True)
    for r in res:
        if r["ok"]:
            ses, seg, k, ci, cj = r["task"]
            np.savez_compressed(vdir / f"bench_ses-0{ses}_trial{seg:02d}_pair{k:02d}.npz", shifts=r["shifts"],
                                centers=r["centers"], ci=ci, cj=cj, seconds=r["seconds"])


# ============================================================================ S2
def read_session_metadata(ses: int):
    import pynwb
    from neurotd.hiEEG_functions import detect_segments
    with pynwb.NWBHDF5IO(str(NWB[ses]), "r", load_namespaces=True) as io:
        nwb = io.read()
        es = nwb.acquisition["ecephys.lfp"]
        ref = np.asarray(es.electrodes.data[:], dtype=int)
        etab = nwb.electrodes
        cols = {c: np.asarray(etab[c].data[:]) for c in ["label", "location", "x", "y", "z", "group_name"]}
        n_rows = len(etab.id.data[:])
        shape = es.data.shape
        dtype = str(es.data.dtype)
        rate = es.rate
        ts = np.asarray(es.timestamps[:], dtype=np.float64)
        tr = nwb.trials
        trials = pd.DataFrame({c: np.asarray(tr[c].data[:]) for c in
                               ["id", "start_time", "stop_time", "artifact", "set_size", "correct",
                                "response_time", "response_type", "solution", "probe_letter"]})
    seg = detect_segments(ts, np.zeros((ts.shape[0], 1), dtype=np.int8))
    seg_t = np.asarray(seg["time"])
    within = np.diff(seg_t, axis=1)
    elec = pd.DataFrame({
        "session": f"ses-0{ses}", "lfp_column": np.arange(ref.size), "electrode_table_row": ref,
        "label": [dec(cols["label"][i]) for i in ref], "location": [dec(cols["location"][i]) for i in ref],
        "x": cols["x"][ref], "y": cols["y"][ref], "z": cols["z"][ref],
        "group_name": [dec(cols["group_name"][i]) for i in ref]})
    elec["major_region"] = elec["location"].map(major_region)
    elec["lateral"] = elec["location"].map(lateral_of)
    first32 = [dec(cols["label"][i]) for i in range(min(32, n_rows))]
    info = dict(session=f"ses-0{ses}", data_shape=shape, data_dtype=dtype, es_rate_attr=rate,
                n_electrode_table_rows=n_rows, n_lfp_electrode_refs=int(ref.size),
                lfp_refs_equal_first_32_rows=bool(np.array_equal(ref, np.arange(ref.size))),
                labels_first_32_table_rows_equal_lfp_labels=first32 == elec["label"].tolist(),
                n_segments=int(seg_t.shape[0]), segment_len=int(seg_t.shape[1]),
                within_segment_dt_min=float(within.min()), within_segment_dt_max=float(within.max()),
                sampling_rate_hz=float(1 / np.median(within)))
    return info, elec, trials, seg_t[:, 0], seg_t[:, -1]


def build_pairs(elec: pd.DataFrame):
    regions = elec["major_region"].to_numpy()
    u = np.unique(regions)
    u = u[u != "unspecific"]
    rows = []
    for r in u:
        cols = np.flatnonzero(regions == r)
        if cols.size < 2:
            continue
        for a in range(cols.size - 1):
            for b in range(a + 1, cols.size):
                i, j = int(cols[a]), int(cols[b])
                pi = elec.loc[i, ["x", "y", "z"]].to_numpy(float)
                pj = elec.loc[j, ["x", "y", "z"]].to_numpy(float)
                rows.append(dict(region=r, channel_i=i, channel_j=j, label_i=elec.label[i], label_j=elec.label[j],
                                 lateral_i=elec.lateral[i], lateral_j=elec.lateral[j],
                                 location_i=elec.location[i], location_j=elec.location[j],
                                 distance_mm=float(np.linalg.norm(pi - pj))))
    return pd.DataFrame(rows)


def stage_s2():
    infos, elecs, trials_all, pairs_all = [], [], [], []
    REP.set_progress(done=0, total=3, unit="sessions")
    for s in (1, 2, 3):
        REP.set_task(f"Reading metadata and timestamps from {NWB[s].name} with pynwb")
        info, elec, trials, seg_start, seg_end = read_session_metadata(s)
        # match LFP segments to trial rows by start time
        starts = trials["start_time"].to_numpy(float)
        rows = []
        used = set()
        for k, s0 in enumerate(seg_start):
            j = int(np.argmin(np.abs(starts - s0)))
            dt = abs(starts[j] - s0)
            if dt > 1e-4 or j in used:
                raise RuntimeError(f"ses-0{s} segment {k} has no unique matching trial (dt={dt})")
            used.add(j)
            t = trials.iloc[j]
            rows.append(dict(session=f"ses-0{s}", session_num=s, trial_pos=k, trial_row=j, trial_id=int(t["id"]),
                             segment_start_s=float(s0), segment_last_sample_s=float(seg_end[k]),
                             trial_start_time=float(t.start_time), trial_stop_time=float(t.stop_time),
                             start_diff_s=float(dt), stop_minus_last_sample_s=float(t.stop_time - seg_end[k]),
                             artifact=bool(t.artifact), set_size=int(t.set_size), correct=bool(t.correct),
                             response_time=float(t.response_time), response_type=dec(t.response_type),
                             solution=dec(t.solution), probe_letter=dec(t.probe_letter)))
        tdf = pd.DataFrame(rows)
        info["segments_in_time_order"] = bool(np.all(np.diff(seg_start) > 0))
        info["trial_row_equals_segment_index"] = bool((tdf.trial_row == tdf.trial_pos).all())
        info["n_trial_table_rows"] = len(trials)
        info["n_artifact"] = int(tdf.artifact.sum())
        pairs = build_pairs(elec)
        pairs.insert(0, "session", f"ses-0{s}")
        pairs.insert(1, "pair_index", np.arange(len(pairs)))
        infos.append(info)
        elecs.append(elec)
        trials_all.append(tdf)
        pairs_all.append(pairs)
        reg_n = elec[elec.major_region != "unspecific"].groupby("major_region").size().to_dict()
        REP.line(f"ses-0{s}: data {info['data_shape']} {info['data_dtype']} | sampling rate {info['sampling_rate_hz']:.3f} Hz | "
                 f"{info['n_segments']} segments × {info['segment_len']} samples | electrode table {info['n_electrode_table_rows']} rows, LFP references "
                 f"{info['n_lfp_electrode_refs']} (equal to first 32 rows: {info['lfp_refs_equal_first_32_rows']}) | "
                 f"artifact-flagged trials {info['n_artifact']} | channels by region {reg_n} | within-region pairs {len(pairs)}")
        REP.set_progress(done=s)
    REP.set_task(None)
    elec_df = pd.concat(elecs, ignore_index=True)
    trial_df = pd.concat(trials_all, ignore_index=True)
    pair_df = pd.concat(pairs_all, ignore_index=True)
    elec_df.to_csv(TAB / "electrode_mapping_all_32_channels.csv", index=False)
    trial_df.to_csv(TAB / "trial_metadata.csv", index=False)
    pair_df.to_csv(TAB / "channel_pair_mapping.csv", index=False)
    pd.DataFrame(infos).to_csv(TAB / "nwb_structure_check.csv", index=False)

    # pairs identical across sessions?
    keycols = ["region", "channel_i", "channel_j", "label_i", "label_j"]
    same_layout = all(pairs_all[0][keycols].equals(p[keycols]) for p in pairs_all[1:])
    REP.line(f"Channel-pair layouts identical across three sessions: {same_layout}")

    # ---- verify old checkpoints
    REP.line("Checking earlier checkpoints: input hashes, channel/region mapping, trial IDs and artifacts, shapes, integer grid, and window centers.")
    state = load_state()
    centers = expected_centers()
    verdict = []
    total_trials = len(trial_df)
    REP.set_progress(done=0, total=total_trials, unit="trials")
    for n, row in enumerate(trial_df.itertuples()):
        s = row.session_num
        p = OLD_CKPT / f"sub-01_ses-0{s}_trial{row.trial_pos:02d}.npz"
        probs = []
        if not state.get("nwb_hash_all_match", False):
            probs.append("NWB hash mismatch vs 2026-10-04")
        if not p.exists():
            probs.append("missing")
        else:
            z = np.load(p, allow_pickle=False)
            pr = pairs_all[s - 1]
            sh = z["shifts_samples"]
            if sh.shape != (len(pr), centers.size):
                probs.append(f"shape {sh.shape}")
            if not np.array_equal(z["channel_i"], pr.channel_i.to_numpy()) or \
                    not np.array_equal(z["channel_j"], pr.channel_j.to_numpy()):
                probs.append("channel mapping")
            if list(z["region"]) != pr.region.tolist():
                probs.append("region labels")
            if int(z["trial_row"]) != row.trial_row or int(z["trial_id"]) != row.trial_id:
                probs.append("trial row/id")
            if bool(z["artifact"]) != row.artifact:
                probs.append("artifact flag")
            if not np.array_equal(z["window_centers"], centers):
                probs.append("window centers")
            if np.abs(sh.astype(int)).max() > MAX_SHIFT:
                probs.append("shift outside search grid")
            if int(z["n_nonfinite_input_samples"]) != 0:
                probs.append("nonfinite input recorded")
        verdict.append(dict(session=row.session, trial_pos=row.trial_pos, file=p.name,
                            passed=not probs, problems="; ".join(probs)))
        REP.set_progress(done=n + 1)
    vdf = pd.DataFrame(verdict)
    n_pass = int(vdf.passed.sum())
    REP.line(f"Checkpoint verification: {n_pass} trial checkpoints verified ({n_pass * 60} pairs), {n_pass} can be skipped; "
             f"{total_trials - n_pass} remain")
    if n_pass < total_trials:
        REP.line("Failed: " + "; ".join(f"{r.session} t{r.trial_pos}: {r.problems}" for r in vdf[~vdf.passed].itertuples()))

    # ---- sample re-estimation (pre-specified, seed 20261031)
    rng = np.random.default_rng(SEED)
    full_trial = int(rng.integers(0, total_trials))
    ft = trial_df.iloc[full_trial]
    tasks = [(int(ft.session_num), int(ft.trial_pos), k, int(r.channel_i), int(r.channel_j))
             for k, r in enumerate(pairs_all[int(ft.session_num) - 1].itertuples())]
    for s in (1, 2, 3):
        pr = pairs_all[s - 1]
        for reg in sc.REGIONS:
            ks = np.flatnonzero(pr.region.to_numpy() == reg)
            for t in rng.choice(50, size=3, replace=False):
                k = int(rng.choice(ks))
                tasks.append((s, int(t), k, int(pr.channel_i[k]), int(pr.channel_j[k])))
    vdir = CK / "reestimate"
    vdir.mkdir(exist_ok=True)

    def ck_name(t):
        return vdir / f"ses-0{t[0]}_trial{t[1]:02d}_pair{t[2]:02d}.npz"

    bench = {(1, 0, k) for k in range(6)}
    rows = []
    pending = []
    for t in tasks:
        f = ck_name(t)
        bf = vdir / f"bench_ses-0{t[0]}_trial{t[1]:02d}_pair{t[2]:02d}.npz"
        src = f if f.exists() else (bf if (t[:3] in bench and bf.exists()) else None)
        if src is None:
            pending.append(t)
        else:
            rows.append((t, np.load(src)["shifts"], "resumed"))
    REP.line(f"[Re-estimation check] Prespecified {len(tasks)} pairs (all 60 pairs of {ft.session} trial {ft.trial_pos} plus 36 random pairs); "
             f"{len(rows)} can reuse checkpoints; {len(pending)} require calculation")

    def compare(t, shifts):
        z = np.load(OLD_CKPT / f"sub-01_ses-0{t[0]}_trial{t[1]:02d}.npz")
        old = z["shifts_samples"][t[2]].astype(int)
        return bool(np.array_equal(np.asarray(shifts).astype(int), old)), int(np.sum(np.asarray(shifts) != old))

    results = []
    for t, shifts, how in rows:
        eq, nd = compare(t, shifts)
        results.append(dict(session=f"ses-0{t[0]}", trial_pos=t[1], pair_index=t[2], channel_i=t[3], channel_j=t[4],
                            region=pairs_all[t[0] - 1].region[t[2]], identical=eq, n_window_diff=nd, source=how))

    def on_done(res):
        t = res["task"]
        sh = res["shifts"]
        ok_c = np.array_equal(res["centers"], expected_centers()) and sh.size == 508
        eq, nd = compare(t, sh)
        np.savez_compressed(ck_name(t), shifts=sh, centers=res["centers"], ci=t[3], cj=t[4], seconds=res["seconds"])
        REP.event("checkpoint_saved", file=str(ck_name(t).relative_to(OUT)), ok=True)
        results.append(dict(session=f"ses-0{t[0]}", trial_pos=t[1], pair_index=t[2], channel_i=t[3], channel_j=t[4],
                            region=pairs_all[t[0] - 1].region[t[2]], identical=eq and ok_c, n_window_diff=nd,
                            source="computed_now", seconds=round(res["seconds"], 2)))
        return eq and ok_c

    n_ok = n_fail = 0
    if pending:
        n_ok, n_fail, _ = run_pair_tasks(pending, "Sampled re-estimation check", on_done, reused=0)
    rdf = pd.DataFrame(results).sort_values(["session", "trial_pos", "pair_index"])
    rdf.to_csv(TAB / "checkpoint_reestimation_consistency.csv", index=False)
    n_ident = int(rdf.identical.sum())
    REP.line(f"[Re-estimation check] {n_ident}/{len(rdf)} pairs match in every window (508 windows/pair); "
             f"{len(pending)} pairs computed this run, {n_fail} failed checks")
    reuse_ok = bool(n_ident == len(rdf) and n_pass == total_trials and same_layout)
    save_state(checkpoint_trials_verified=n_pass, checkpoint_trials_total=total_trials,
               reestimate_tasks=len(tasks), reestimate_identical=n_ident, reestimate_computed_now=len(pending),
               reestimate_failed=n_fail, reuse_old_checkpoints=reuse_ok,
               failed_trials=vdf[~vdf.passed][["session", "trial_pos"]].values.tolist() if not reuse_ok else [],
               n_pairs_per_trial=len(pairs_all[0]), same_layout=same_layout)
    vdf.to_csv(TAB / "old_checkpoint_verification.csv", index=False)
    if not reuse_ok:
        REP.line("Some earlier checkpoints failed verification. S3 will re-estimate those trials with the original estimator.")


# ============================================================================ S3
def stage_s3():
    st = load_state()
    trial_df = pd.read_csv(TAB / "trial_metadata.csv")
    pair_df = pd.read_csv(TAB / "channel_pair_mapping.csv")
    n_pairs = st["n_pairs_per_trial"]
    total_pairs = len(trial_df) * n_pairs
    new_dir = CK / "recomputed_trials"
    new_dir.mkdir(exist_ok=True)
    failed = set()
    if not st.get("reuse_old_checkpoints"):
        vdf = pd.read_csv(TAB / "old_checkpoint_verification.csv")
        failed = {(r.session, int(r.trial_pos)) for r in vdf[~vdf.passed].itertuples()}
        if not st.get("nwb_hash_all_match", True) or not st.get("same_layout", True) or \
                st.get("reestimate_identical", 0) < st.get("reestimate_tasks", 0):
            failed = {(r.session, int(r.trial_pos)) for r in trial_df.itertuples()}
    # resume: recomputed trial checkpoints from an earlier attempt of this run
    done_new = {(f"ses-0{p.name[5]}", int(p.stem.split("trial")[1])) for p in new_dir.glob("ses-0*_trial*.npz")}
    todo = sorted(failed - done_new)
    reused_pairs = (len(trial_df) - len(failed)) * n_pairs
    REP.line(f"Verified totals: {len(trial_df)} trials × {n_pairs} pairs = {total_pairs} trial/pair records, "
             f"{total_pairs * 508} windows")
    REP.line(f"Resuming: {len(failed)} trials require re-estimation, {len(failed & done_new)} already complete, {len(todo)} remain")
    if not todo:
        REP.line(f"[Delay estimates]\nNew pairs: 0 | reused {reused_pairs} pairs ({reused_pairs * 508} windows from verified earlier checkpoints)\n"
                 f"Re-estimation check (reported separately in S2): {st.get('reestimate_identical')}/{st.get('reestimate_tasks')} pairs agree; "
                 f"{st.get('reestimate_computed_now')} pairs computed this run, {st.get('reestimate_failed')} failures\n"
                 f"Skipped {total_pairs} pair computations, about {fmt_hms(total_pairs * st.get('per_pair_wall_s', np.nan))} at measured throughput")
    n_new_ok = n_new_fail = 0
    for ses, pos in todo:
        s = int(ses[-1])
        pr = pair_df[pair_df.session == ses]
        tasks = [(s, pos, k, int(r.channel_i), int(r.channel_j)) for k, r in enumerate(pr.itertuples())]
        out = {}

        def on_done(res):
            sh = res["shifts"]
            ok = sh.size == 508 and np.all(np.abs(sh) <= MAX_SHIFT) and np.all(sh == np.rint(sh))
            if ok:
                out[res["task"][2]] = sh.astype(np.int16)
            return ok
        REP.line(f"[Delay estimates] session {s}/3 | trial {pos + 1}/50 | {len(tasks)} pairs to compute")
        ok, fail, _ = run_pair_tasks(tasks, "Delay estimation", on_done, reused=reused_pairs)
        n_new_ok += ok
        n_new_fail += fail
        if fail == 0:
            path = new_dir / f"{ses}_trial{pos:02d}.npz"
            np.savez_compressed(path, shifts_samples=np.stack([out[k] for k in range(len(tasks))]))
            REP.line(f"Saved checkpoint {path.relative_to(OUT)}")
        else:
            REP.line(f"Checkpoint: {ses} trial {pos} had {fail} failed pairs; retry before saving.")
    # consolidate window array
    T = len(trial_df)
    Wv = np.zeros((T, n_pairs, 508), dtype=np.int16)
    src = []
    for t, r in enumerate(trial_df.itertuples()):
        newp = new_dir / f"{r.session}_trial{r.trial_pos:02d}.npz"
        if (r.session, r.trial_pos) in failed:
            if not newp.exists():
                raise RuntimeError(f"{r.session} trial {r.trial_pos} has no available delay result")
            Wv[t] = np.load(newp)["shifts_samples"]
            src.append("recomputed_this_run")
        else:
            Wv[t] = np.load(OLD_CKPT / f"sub-01_{r.session}_trial{r.trial_pos:02d}.npz")["shifts_samples"]
            src.append("reused_2026_10_04_checkpoint_verified")
    np.savez_compressed(CK / "windows_all_trials.npz", shifts_samples=Wv,
                        session=trial_df.session.to_numpy().astype(str), trial_pos=trial_df.trial_pos.to_numpy(),
                        window_centers=expected_centers())
    REP.line(f"Saved checkpoints/windows_all_trials.npz, shape {Wv.shape}")
    save_state(pairs_reused=reused_pairs, pairs_recomputed=n_new_ok, pairs_recompute_failed=n_new_fail,
               trial_sources=pd.Series(src).value_counts().to_dict(), total_pair_trials=total_pairs,
               total_windows=int(Wv.size))


# ============================================================================ S4
def stage_s4():
    trial_df = pd.read_csv(TAB / "trial_metadata.csv")
    pair_df = pd.read_csv(TAB / "channel_pair_mapping.csv")
    z = np.load(CK / "windows_all_trials.npz")
    Wv = z["shifts_samples"].astype(np.int32)
    pr = pair_df[pair_df.session == "ses-01"].reset_index(drop=True)
    reg = pr.region.to_numpy()
    ms = Wv * MS_PER_SAMPLE
    absms = np.abs(ms)
    nz = absms > NEAR_ZERO
    fin = np.isfinite(ms)
    rows, prow = [], []
    REP.set_progress(done=0, total=len(trial_df), unit="trials")
    for t, tr in enumerate(trial_df.itertuples()):
        for k in range(len(pr)):
            a = absms[t, k][nz[t, k]]
            prow.append(dict(session=tr.session, trial_pos=tr.trial_pos, trial_id=tr.trial_id, region=reg[k],
                             pair_index=k, channel_i=pr.channel_i[k], channel_j=pr.channel_j[k],
                             label_i=pr.label_i[k], label_j=pr.label_j[k], n_windows_raw=int(fin[t, k].sum()),
                             n_zero=int((fin[t, k] & ~nz[t, k]).sum()), N_nonzero=int(a.size), S_nonzero_ms=float(a.sum()),
                             mean_nonzero_ms=float(a.mean()) if a.size else np.nan,
                             median_nonzero_ms=float(np.median(a)) if a.size else np.nan,
                             n_boundary_500=int((np.abs(Wv[t, k]) == MAX_SHIFT).sum())))
        for r in sc.REGIONS:
            m = reg == r
            blk = absms[t, m]
            a = blk[nz[t, m]]
            allf = blk[fin[t, m]]
            q = np.quantile(a, [0.25, 0.5, 0.75]) if a.size else [np.nan] * 3
            rows.append(dict(session=tr.session, session_num=tr.session_num, trial_pos=tr.trial_pos,
                             trial_row=tr.trial_row, trial_id=tr.trial_id, region=r, n_pairs=int(m.sum()),
                             n_windows_raw=int(allf.size), n_zero=int(allf.size - a.size), N_nonzero=int(a.size),
                             S_nonzero_ms=float(a.sum()), mean_nonzero_ms=float(a.mean()) if a.size else np.nan,
                             median_nonzero_ms=float(q[1]), q25_nonzero_ms=float(q[0]), q75_nonzero_ms=float(q[2]),
                             mean_including_zero_ms=float(allf.mean()), median_including_zero_ms=float(np.median(allf)),
                             n_boundary_500=int((np.abs(Wv[t, m]) == MAX_SHIFT).sum()),
                             artifact=tr.artifact, set_size=tr.set_size, correct=tr.correct,
                             response_time=tr.response_time, trial_start_time=tr.trial_start_time,
                             trial_stop_time=tr.trial_stop_time))
        REP.set_progress(done=t + 1)
    trs = pd.DataFrame(rows)
    pts = pd.DataFrame(prow)
    trs.to_csv(TAB / "trial_region_summary.csv", index=False)
    pts.to_csv(TAB / "trial_pair_summary.csv", index=False)
    # reconstruction check: pooled means from S,N equal the direct window mean
    chk = []
    for r in sc.REGIONS:
        m = reg == r
        direct = absms[:, m][nz[:, m]]
        sub = trs[trs.region == r]
        chk.append(dict(region=r, direct_window_mean=float(direct.mean()),
                        pooled_from_S_N=float(sub.S_nonzero_ms.sum() / sub.N_nonzero.sum()),
                        abs_diff=float(abs(direct.mean() - sub.S_nonzero_ms.sum() / sub.N_nonzero.sum())),
                        N_direct=int(direct.size), N_from_table=int(sub.N_nonzero.sum())))
    cdf = pd.DataFrame(chk)
    cdf.to_csv(TAB / "check_summary_reproduces_pooled_means.csv", index=False)
    REP.line("Pooled means recovered from summary: " + "; ".join(f"{sc.REGION_SHORT[r.region]} {r.pooled_from_S_N:.6f} ms "
                                         f"(|difference| {r.abs_diff:.1e}, counts agree {r.N_direct == r.N_from_table})"
                                         for r in cdf.itertuples()))
    nmiss = int((pts.N_nonzero == 0).sum())
    REP.line(f"Trial/region rows {len(trs)}; trial/channel-pair rows {len(pts)}; trial/pairs without nonzero windows {nmiss}; "
             f"windows at |shift|=500 boundary {int(trs.n_boundary_500.sum())}")
    save_state(n_trial_pair_no_nonzero=nmiss)


# ============================================================================ S5
def stage_s5():
    from scipy.stats import ttest_ind
    import statsmodels.stats.multitest as smm
    pair_df = pd.read_csv(TAB / "channel_pair_mapping.csv")
    reg = pair_df[pair_df.session == "ses-01"].region.to_numpy()
    z = np.load(CK / "windows_all_trials.npz")
    absms = np.abs(z["shifts_samples"].astype(np.int32) * MS_PER_SAMPLE)
    sess = z["session"]
    desc, welch = [], []
    pooled_vals = {}
    for scope in ["pooled_3_sessions", "ses-01", "ses-02", "ses-03"]:
        tm = np.ones(len(sess), bool) if scope.startswith("pooled") else (sess == scope)
        vals = {}
        for r in sc.REGIONS:
            blk = absms[tm][:, reg == r].ravel()
            v = blk[blk > NEAR_ZERO]
            vals[r] = v
            q = np.quantile(v, [0.05, 0.25, 0.5, 0.75, 0.95])
            desc.append(dict(scope=scope, region=r, n_windows_raw=int(blk.size), n_zero=int(blk.size - v.size),
                             zero_fraction=float(1 - v.size / blk.size), n_nonzero=int(v.size),
                             mean_ms=float(v.mean()), sd_ms=float(v.std(ddof=1)), q05=q[0], q25=q[1], median_ms=q[2],
                             q75=q[3], q95=q[4], n_at_boundary_15_625ms=int((np.abs(v - 15.625) < 1e-9).sum())))
        res = []
        for a, b in sc.COMPARISONS:
            ra, rb = sc.REGIONS[a], sc.REGIONS[b]
            t, p = ttest_ind(vals[ra], vals[rb], equal_var=False)
            res.append(dict(scope=scope, region_A=ra, region_B=rb, mean_A=vals[ra].mean(), mean_B=vals[rb].mean(),
                            difference_ms=vals[ra].mean() - vals[rb].mean(), welch_t=float(t), raw_p=float(p),
                            n_A=vals[ra].size, n_B=vals[rb].size))
        adj = smm.multipletests([r["raw_p"] for r in res], method="bonferroni")[1]
        for r, pa in zip(res, adj):
            r["bonferroni_p"] = float(pa)
            r["stars_old_rule"] = sc.stars(pa)
            r["note"] = "window-level Welch: overlapping, dependent windows treated as independent; audit only, not valid inference"
        welch += res
        if scope.startswith("pooled"):
            pooled_vals = vals
    ddf = pd.DataFrame(desc)
    wdf = pd.DataFrame(welch)
    ddf.to_csv(TAB / "window_level_descriptives_by_scope.csv", index=False)
    wdf.to_csv(TAB / "window_level_welch_audit_only.csv", index=False)
    ref = []
    for r in sc.REGIONS:
        med = float(np.median(pooled_vals[r]))
        ref.append(dict(item=f"median {r}", historical=HIST_MEDIANS[r], this_run=med, match=med == HIST_MEDIANS[r]))
    tot = int(sum(v.size for v in pooled_vals.values()))
    ref.append(dict(item="total nonzero windows", historical=HIST_TOTAL_NONZERO, this_run=tot,
                    match=tot == HIST_TOTAL_NONZERO))
    rdf = pd.DataFrame(ref)
    rdf.to_csv(TAB / "historical_reference_check.csv", index=False)
    for r in rdf.itertuples():
        REP.line(f"Reference check {r.item}: historical {r.historical} | current {r.this_run} | match {r.match}")
    for r in wdf[wdf.scope == "pooled_3_sessions"].itertuples():
        REP.line(f"Window-level Welch (audit only) {sc.REGION_SHORT[r.region_A]}-{sc.REGION_SHORT[r.region_B]}: "
                 f"Δ={r.difference_ms:+.4f} ms t={r.welch_t:.2f} Bonferroni p={r.bonferroni_p:.3g}")
    # subsample for violin display, saved for re-plotting
    rng = np.random.default_rng(0)
    sub = {sc.REGION_SHORT[r]: rng.choice(v, size=min(200_000, v.size), replace=False) for r, v in pooled_vals.items()}
    np.savez_compressed(FIG / "source_data/fig1_violin_window_subsample_200k_per_region_seed0.npz", **sub)


# ============================================================================ S6
def load_trial_arrays():
    trs = pd.read_csv(TAB / "trial_region_summary.csv")
    pts = pd.read_csv(TAB / "trial_pair_summary.csv")
    meta = pd.read_csv(TAB / "trial_metadata.csv")
    T = len(meta)
    key = meta.session + "_" + meta.trial_pos.astype(str)
    order = {k: i for i, k in enumerate(key)}
    S = np.zeros((T, 4)); N = np.zeros((T, 4)); Nall = np.zeros((T, 4)); MED = np.full((T, 4), np.nan)
    for r in trs.itertuples():
        t = order[f"{r.session}_{r.trial_pos}"]
        j = sc.REGIONS.index(r.region)
        S[t, j], N[t, j], Nall[t, j] = r.S_nonzero_ms, r.N_nonzero, r.n_windows_raw
        MED[t, j] = r.median_nonzero_ms if r.N_nonzero > 0 else np.nan
    V = np.full((T, 4), np.nan)
    pts["t"] = [order[f"{s}_{p}"] for s, p in zip(pts.session, pts.trial_pos)]
    for (t, r), g in pts.groupby(["t", "region"]):
        m = g.mean_nonzero_ms.to_numpy()
        m = m[np.isfinite(m)]
        V[t, sc.REGIONS.index(r)] = m.mean() if m.size else np.nan
    return meta, S, N, Nall, MED, V


def design_matrix(meta):
    T = len(meta)
    cols, names = [], []
    for s in (1, 2, 3):
        cols.append((meta.session_num == s).astype(float).to_numpy()); names.append(f"session_{s}")
    posc = meta.trial_pos.to_numpy(float) - 24.5
    for s in (1, 2, 3):
        cols.append(posc * (meta.session_num == s)); names.append(f"trend_ses{s}")
    levels = sorted(meta.set_size.unique())
    for lv in levels[1:]:
        c = (meta.set_size == lv).astype(float).to_numpy()
        cols.append(c - c.mean()); names.append(f"set_size_{lv}")
    c = meta.correct.astype(float).to_numpy()
    if c.std() > 0:
        cols.append(c - c.mean()); names.append("correct")
    return np.column_stack(cols), names


def compute_stats(C, S, N, Nall, MED, V, X, w_all, w_art, masks, variants):
    out = {}
    D6_trialmeans = None
    for v in variants:
        if v == "main_pooled_nonzero_all_trials":
            out[v] = sc.pairwise_diffs(sc.pooled_means(C, S, N, w_all))
        elif v == "A_pooled_including_zero":
            out[v] = sc.pairwise_diffs(sc.pooled_means(C, S, Nall, w_all))
        elif v == "B_pooled_artifact_excluded":
            out[v] = sc.pairwise_diffs(sc.pooled_means(C, S, N, w_art))
        elif v == "D_pair_trial_session_equal_weight":
            out[v] = sc.pairwise_diffs(sc.session_equal_means(C, V, w_all, masks))
        elif v == "E_trial_median_difference":
            out[v] = sc.session_equal_means(C, sc.pairwise_diffs(MED), w_all, masks)
        elif v == "F_trend_condition_adjusted":
            if D6_trialmeans is None:
                with np.errstate(invalid="ignore", divide="ignore"):
                    D6_trialmeans = sc.pairwise_diffs(np.where(N > 0, S / N, np.nan))
            out[v], rd = sc.adjusted_regression(C, D6_trialmeans, X, 3)
            out["_F_rank_def"] = rd
    return out


def run_synthetic_validation():
    """Checks of the resampling program on synthetic inputs; saved separately from real results."""
    rows = []

    def rec(name, ok, detail):
        rows.append(dict(check=name, passed=bool(ok), detail=detail))
        REP.line(f"  [Synthetic check] {name}: {'pass' if ok else 'fail'} | {detail}")

    rng = np.random.default_rng(123)
    T = 150
    slices = [(0, 50), (50, 100), (100, 150)]
    masks = [np.r_[np.zeros(a, bool), np.ones(b - a, bool), np.zeros(T - b, bool)] for a, b in slices]
    # 1 no cross-session draws and circular blocks inside each session
    for L in LS:
        idx = sc.stratified_indices(np.random.default_rng(5), slices, L, 2000)
        ok = all(((idx[:, i * 50:(i + 1) * 50] >= a) & (idx[:, i * 50:(i + 1) * 50] < b)).all()
                 for i, (a, b) in enumerate(slices))
        loc = idx[:, :50]
        steps = (loc[:, 1:] - loc[:, :-1]) % 50
        inblock = np.ones(49, bool)
        inblock[np.arange(L - 1, 49, L)] = False
        ok2 = (steps[:, inblock] == 1).all()
        rec(f"L={L} sampling stays within session; each circular block is contiguous", ok and ok2, f"2000 draws; all within-block adjacent steps are +1 (mod 50): {ok2}")
    # 2 same seed reproducible
    a = sc.stratified_indices(np.random.default_rng(np.random.SeedSequence([SEED, 0, 5])), slices, 5, 1000)
    b = sc.stratified_indices(np.random.default_rng(np.random.SeedSequence([SEED, 0, 5])), slices, 5, 1000)
    rec("Same seed reproduces the same indices", np.array_equal(a, b), "Both index arrays match element by element")
    # 3 identical regions -> zero difference in every resample; same index for all regions
    S1 = rng.gamma(5, 2, (T, 1)) * 1000
    N1 = rng.integers(500, 1500, (T, 1)).astype(float)
    S = np.repeat(S1, 4, axis=1); N = np.repeat(N1, 4, axis=1)
    C = sc.counts_from_indices(a, T)
    d = sc.pairwise_diffs(sc.pooled_means(C, S, N, np.ones(T)))
    dh = sc.pairwise_diffs(sc.pooled_means(np.ones((1, T)), S, N, np.ones(T)))[0]
    summ = sc.summarize(dh, d)
    rec("Identical regional inputs give zero contrast on every draw (all four regions share each trial index)", np.all(d == 0) and np.all(dh == 0),
        f"max|Δ*|={np.abs(d).max()}; p_raw={summ[0]['raw_p']} (p=1 when Δ_hat=0)")
    # 4 swapping order flips the sign only
    S = rng.gamma(5, 2, (T, 4)) * 1000
    N = rng.integers(500, 1500, (T, 4)).astype(float)
    mu = sc.pooled_means(C, S, N, np.ones(T))
    d_ab = mu[:, 0] - mu[:, 1]
    d_ba = mu[:, 1] - mu[:, 0]
    mh = sc.pooled_means(np.ones((1, T)), S, N, np.ones(T))[0]
    s_ab = sc.summarize(np.array([mh[0] - mh[1]]), d_ab[:, None])[0]
    s_ba = sc.summarize(np.array([mh[1] - mh[0]]), d_ba[:, None])[0]
    rec("Reversing contrast order reverses only the sign", np.allclose(d_ab, -d_ba) and s_ab["raw_p"] == s_ba["raw_p"] and
        np.isclose(s_ab["ci95_low"], -s_ba["ci95_high"]), f"p {s_ab['raw_p']:.4g} vs {s_ba['raw_p']:.4g}")
    # 5 zero counts / all-zero region / missing values
    N0 = N.copy(); S0 = S.copy()
    N0[3, 1] = 0; S0[3, 1] = 0
    m1 = sc.pooled_means(np.ones((1, T)), S0, N0, np.ones(T))[0, 1]
    exp = S0[:, 1].sum() / N0[:, 1].sum()
    N0[:, 2] = 0; S0[:, 2] = 0
    m2 = sc.pooled_means(np.ones((1, T)), S0, N0, np.ones(T))[0, 2]
    rec("Trials with zero counts contribute nothing; a regional N=0 gives a missing mean", np.isclose(m1, exp) and np.isnan(m2),
        f"Pooled mean with one zero-count trial {m1:.4f}, expected {exp:.4f}; all-zero region gives {m2}")
    Vm = S / N
    Vm[0:50, 0] = np.nan
    val = sc.session_equal_means(np.ones((1, T)), Vm, np.ones(T), masks)[0, 0]
    rec("An entirely missing region in a session gives a missing equal-weight estimate and an invalid draw", np.isnan(val), f"value={val}")
    Vm = S / N
    Vm[7, 0] = np.nan
    val = sc.session_equal_means(np.ones((1, T)), Vm, np.ones(T), masks)[0, 0]
    expv = np.mean([np.nanmean(Vm[0:50, 0]), Vm[50:100, 0].mean(), Vm[100:150, 0].mean()])
    rec("A single missing value is skipped", np.isclose(val, expv), f"{val:.5f} vs {expv:.5f}")
    # 6 artifact weighting keeps grid and removes contribution
    w = np.ones(T); w[[4, 60]] = 0
    m3 = sc.pooled_means(np.ones((1, T)), S, N, w)[0, 0]
    keep = w > 0
    rec("Zero artifact weight is equivalent to dropping that trial's S and N", np.isclose(m3, S[keep, 0].sum() / N[keep, 0].sum()), f"{m3:.5f}")
    # 7 calibration: null AR(1) trial series, iid vs block bootstrap rejection rate (small, illustrative)
    rej = {1: 0, 5: 0}
    nrep = 300
    rr = np.random.default_rng(77)
    for rep in range(nrep):
        e = np.zeros((T, 2))
        for i, (s0, s1) in enumerate(slices):
            z = rr.normal(size=(s1 - s0,))
            x = np.zeros(s1 - s0)
            for k in range(s1 - s0):
                x[k] = (0.6 * x[k - 1] if k else 0) + z[k]
            e[s0:s1, 0] = x
            e[s0:s1, 1] = rr.normal(size=s1 - s0) * 0.3
        Ss = np.column_stack([1000 * (10 + e[:, 0]), 1000 * (10 + e[:, 1])]); Ns = np.full((T, 2), 1000.0)
        dh = (Ss[:, 0].sum() / Ns[:, 0].sum()) - (Ss[:, 1].sum() / Ns[:, 1].sum())
        for L in (1, 5):
            idx = sc.stratified_indices(rr, slices, L, 999)
            Cb = sc.counts_from_indices(idx, T)
            mu = (Cb @ Ss) / (Cb @ Ns)
            p = sc.summarize(np.array([dh]), (mu[:, 0] - mu[:, 1])[:, None], n_tests=1)[0]["raw_p"]
            rej[L] += p < 0.05
    rec("Illustrative rejection rate at 5% for synthetic AR(1) null series (φ=0.6)", True,
        f"L=1: {rej[1] / nrep:.3f}; L=5: {rej[5] / nrep:.3f} ({nrep} replicates, B=999; dependence can make L=1 liberal)")
    df = pd.DataFrame(rows)
    df.to_csv(SYN / "synthetic_validation_results.csv", index=False)
    return bool(df.passed.all())


def acf_pooled(x, sess, maxlag=15):
    num = np.zeros(maxlag + 1); den = 0.0
    for s in np.unique(sess):
        v = x[sess == s]
        v = np.where(np.isfinite(v), v, np.nanmean(v))
        v = v - v.mean()
        den += np.sum(v * v)
        for k in range(maxlag + 1):
            num[k] += np.sum(v[:len(v) - k] * v[k:])
    return num / den


def stage_s6():
    REP.line("Step 1/4: validate statistical procedures on synthetic data (saved separately in synthetic_validation/)")
    ok = run_synthetic_validation()
    if not ok:
        raise RuntimeError("Synthetic checks failed. Stopping the bootstrap stage.")
    meta, S, N, Nall, MED, V = load_trial_arrays()
    X, xnames = design_matrix(meta)
    T = len(meta)
    w_all = np.ones(T)
    w_art = (~meta.artifact.astype(bool)).astype(float).to_numpy()
    sess_num = meta.session_num.to_numpy()
    slices_of = {s: (int(np.flatnonzero(sess_num == s)[0]), int(np.flatnonzero(sess_num == s)[-1]) + 1) for s in (1, 2, 3)}
    for s, (a, b) in slices_of.items():
        assert b - a == 50 and np.array_equal(meta.trial_pos.to_numpy()[a:b], np.arange(50))
    REP.line(f"Variant F design matrix columns: {xnames}")

    settings = []
    for scope, (code, sessions) in SCOPES.items():
        for L in LS:
            settings.append((scope, code, sessions, L, VARIANTS_POOLED if scope == "pooled_3_sessions" else VARIANTS_SUB))
    n_variant_settings = sum(len(s[4]) for s in settings)
    REP.line(f"Step 2/4: bootstrap {len(settings)} index streams (scope × L) across {n_variant_settings} processing variant × scope × L settings, "
             f"B={B_MAIN} per setting, batch size {BATCH}")
    bdir = CK / "bootstrap"
    bdir.mkdir(exist_ok=True)
    done_files = {p.name for p in bdir.glob("*.npz")}
    names = [f"{sc_}_L{L}_B{B_MAIN}_main.npz" for sc_, _, _, L, _ in settings]
    n_skip = sum(n in done_files for n in names)
    REP.line(f"Resuming: {n_skip} bootstrap checkpoints, {n_skip} can be skipped, {len(settings) - n_skip} remain")

    all_rows = []
    t_stage = time.time()
    rt_set = RateTracker(min_units=2, min_seconds=5)

    def run_setting(scope, code, sessions, L, variants, B, seed_key, tag, k_set, n_set):
        fname = bdir / f"{scope}_L{L}_B{B}_{tag}.npz"
        sl = [slices_of[s] for s in sessions]
        masks = [sess_num == s for s in sessions]
        in_scope = np.isin(sess_num, sessions)
        C_obs = in_scope.astype(float)[None, :]
        obs = compute_stats(C_obs, S, N, Nall, MED, V, X, w_all, w_art, masks, variants)
        if fname.exists():
            zz = np.load(fname)
            boots = {v: zz[v] for v in variants}
            rd = int(zz["F_rank_def"]) if "F_rank_def" in zz.files else 0
            REP.line(f"[bootstrap] Reusing checkpoint {fname.name}")
        else:
            rng = np.random.default_rng(np.random.SeedSequence(seed_key))
            boots = {v: np.empty((B, 6)) for v in variants}
            rd = 0
            t0 = time.time()
            last = 0
            for start in range(0, B, BATCH):
                b = min(BATCH, B - start)
                idx = sc.stratified_indices(rng, sl, L, b)
                C = sc.counts_from_indices(idx, T)
                res = compute_stats(C, S, N, Nall, MED, V, X, w_all, w_art, masks, variants)
                rd += res.pop("_F_rank_def", 0)
                for v in variants:
                    boots[v][start:start + b] = res[v]
                done = start + b
                REP.set_progress(done=done, total=B, unit="replicates",
                                 extra=f"setting {k_set}/{n_set}")
                el = time.time() - t0
                if time.time() - last >= 10 or done == B:
                    last = time.time()
                    eta = el / done * (B - done)
                    REP.line(f"[bootstrap] variant {'+'.join(v.split('_')[0] for v in variants)} (shared indices) | scope {scope} | "
                             f"L={L} | draws {done}/{B} | setting stream {k_set}/{n_set} | elapsed {fmt_hms(el)} | "
                             f"estimated remaining {fmt_hms(eta)}")
            np.savez_compressed(fname, F_rank_def=rd, seed_key=np.array(seed_key), **boots)
            REP.line(f"Saved bootstrap checkpoint checkpoints/bootstrap/{fname.name}")
        rows = []
        for v in variants:
            summ = sc.summarize(obs[v][0], boots[v])
            for kk, (a, bb) in enumerate(sc.COMPARISONS):
                r = summ[kk]
                if v in ("D_pair_trial_session_equal_weight",):
                    ntr = int(np.sum(np.isfinite(V[in_scope][:, a]) & np.isfinite(V[in_scope][:, bb])))
                elif v == "E_trial_median_difference":
                    ntr = int(np.sum(np.isfinite(MED[in_scope][:, a]) & np.isfinite(MED[in_scope][:, bb])))
                elif v == "F_trend_condition_adjusted":
                    ntr = int(np.sum((N[in_scope][:, a] > 0) & (N[in_scope][:, bb] > 0)))
                elif v == "B_pooled_artifact_excluded":
                    ntr = int(w_art[in_scope].sum())
                else:
                    ntr = int(in_scope.sum())
                rows.append(dict(analysis_variant=v, session_scope=scope, region_A=sc.REGIONS[a], region_B=sc.REGIONS[bb],
                                 estimand=ESTIMAND[v], block_length=L, bootstrap_B=B, seed_key=str(list(seed_key)),
                                 run_tag=tag, n_trials_in_scope=int(in_scope.sum()), n_trials_effective=ntr,
                                 F_rank_deficient_resamples=rd if v.startswith("F") else "",
                                 is_primary=(v == VARIANTS_POOLED[0] and scope == "pooled_3_sessions" and L == REF_L and tag == "main"),
                                 **r, significance_bonferroni=sc.stars(r["adjusted_p"])))
        return rows

    for k, (scope, code, sessions, L, variants) in enumerate(settings, start=1):
        REP.set_task(f"{scope} L={L}")
        all_rows += run_setting(scope, code, sessions, L, variants, B_MAIN, [SEED, code, L], "main", k, len(settings))
        REP.line(f"[bootstrap] Completed setting stream {k}/{len(settings)} ({scope}, L={L}) | {rt_set.text(k, len(settings), 'stream')}")
    res = pd.DataFrame(all_rows)

    REP.line("Step 3/4: Monte Carlo precision and prespecified refinement (p_adj within 3×6×SE of 0.05/0.01/0.001)")
    need = res[[sc.near_threshold(p, se) for p, se in zip(res.adjusted_p, res.p_mc_se)]]
    refine_keys = sorted({(r.session_scope, r.block_length) for r in need.itertuples()})
    REP.line(f"Refinement triggered for (scope, L): {refine_keys if refine_keys else 'none'}")
    for k, (scope, L) in enumerate(refine_keys, start=1):
        code, sessions = SCOPES[scope]
        variants = VARIANTS_POOLED if scope == "pooled_3_sessions" else VARIANTS_SUB
        REP.set_task(f"Refine {scope} L={L} B={B_REFINE}")
        all_rows += run_setting(scope, code, sessions, L, variants, B_REFINE, [SEED, code, L, 1], "refine", k, len(refine_keys))
    res = pd.DataFrame(all_rows)
    REP.set_task(None)

    REP.line("Step 4/4: assess block inference (time series, autocorrelation, trends, task conditions)")
    import statsmodels.api as sm
    with np.errstate(invalid="ignore", divide="ignore"):
        Mtr = np.where(N > 0, S / N, np.nan)
    D6 = sc.pairwise_diffs(Mtr)
    band = 1.96 / np.sqrt(T)
    diag_rows, acf_rows, trend_rows, cov_rows = [], [], [], []
    main_res = res[(res.analysis_variant == VARIANTS_POOLED[0]) & (res.run_tag == "main")]
    for kk, (a, b) in enumerate(sc.COMPARISONS):
        name = f"{sc.REGION_SHORT[sc.REGIONS[a]]}-{sc.REGION_SHORT[sc.REGIONS[b]]}"
        d = D6[:, kk]
        acf = acf_pooled(d, sess_num)
        for lag in range(1, 16):
            acf_rows.append(dict(comparison=name, lag=lag, acf=acf[lag], band_95=band, exceeds=abs(acf[lag]) > band))
        n_far = int(sum(abs(acf[l]) > band for l in range(6, 16)))
        flags = []
        n_trend_sig = 0
        trend_big = False
        sess_d = {}
        for s in (1, 2, 3):
            m = sess_num == s
            y = d[m]
            x = sm.add_constant(meta.trial_pos.to_numpy(float)[m])
            fit = sm.OLS(y, x, missing="drop").fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            sd = main_res[(main_res.session_scope == f"ses-0{s}_only") & (main_res.block_length == REF_L) &
                          (main_res.region_A == sc.REGIONS[a]) & (main_res.region_B == sc.REGIONS[b])]
            sdelta = float(sd.difference_ms.iloc[0])
            sess_d[s] = sd.iloc[0]
            span = fit.params[1] * 49
            trend_rows.append(dict(comparison=name, session=f"ses-0{s}", slope_ms_per_trial=fit.params[1],
                                   hac_se=fit.bse[1], hac_p=fit.pvalues[1], change_over_session_ms=span,
                                   session_pooled_delta_ms=sdelta,
                                   ratio_change_to_delta=abs(span) / abs(sdelta) if sdelta else np.nan,
                                   spearman_rho=pd.Series(y).corr(pd.Series(np.arange(50.0)), method="spearman")))
            n_trend_sig += fit.pvalues[1] < 0.05
            trend_big |= abs(span) > 0.5 * abs(sdelta)
        if n_trend_sig >= 2 or trend_big:
            flags.append(f"Trend warning (sessions with HAC p<0.05: {n_trend_sig}; within-session change > 0.5|Δ|: {trend_big})")
        se5 = main_res[(main_res.session_scope == "pooled_3_sessions") & (main_res.block_length == 5)].boot_se.to_numpy()[kk]
        se10 = main_res[(main_res.session_scope == "pooled_3_sessions") & (main_res.block_length == 10)].boot_se.to_numpy()[kk]
        se1 = main_res[(main_res.session_scope == "pooled_3_sessions") & (main_res.block_length == 1)].boot_se.to_numpy()[kk]
        if n_far >= 2 or se10 > 1.2 * se5:
            flags.append(f"Dependence may exceed block length ({n_far} lags 6–15 outside the band; SE(L10)/SE(L5)={se10 / se5:.2f})")
        signs = {np.sign(sess_d[s].difference_ms) for s in (1, 2, 3)}
        overlap = all(max(sess_d[i].ci95_low, sess_d[j].ci95_low) <= min(sess_d[i].ci95_high, sess_d[j].ci95_high)
                      for i, j in [(1, 2), (1, 3), (2, 3)])
        if len(signs) > 1 or not overlap:
            flags.append(f"Session heterogeneity (directions agree: {len(signs) == 1}; pairwise overlap of 95% intervals: {overlap})")
        diag_rows.append(dict(comparison=name, region_A=sc.REGIONS[a], region_B=sc.REGIONS[b], acf_lag1=acf[1], acf_lag2=acf[2],
                              acf_lag5=acf[5], n_lags_6_15_beyond_band=n_far, boot_se_L1=se1, boot_se_L5=se5,
                              boot_se_L10=se10, se_ratio_L10_L5=se10 / se5, se_ratio_L5_L1=se5 / se1,
                              n_sessions_trend_hac_p_lt_05=n_trend_sig, trend_change_gt_half_delta=trend_big,
                              session_signs_consistent=len(signs) == 1, session_ci95_pairwise_overlap=overlap,
                              inference_condition_flags=" | ".join(flags) if flags else "No prespecified warning triggered; independence and stationarity remain assumptions"))
        # covariates: descriptive regression with panel HAC
        dfc = pd.DataFrame({"d": d, "session": sess_num, "pos": meta.trial_pos, "set_size": meta.set_size.astype(str),
                            "correct": meta.correct.astype(int), "artifact": meta.artifact.astype(int)}).dropna()
        import statsmodels.formula.api as smf
        fit = smf.ols("d ~ C(session) + pos:C(session) + C(set_size) + correct + artifact", data=dfc).fit(
            cov_type="hac-panel", cov_kwds={"groups": dfc.session, "time": dfc.pos, "maxlags": 5})
        for term in fit.params.index:
            if term.startswith(("C(set_size)", "correct", "artifact")):
                cov_rows.append(dict(comparison=name, term=term, coef_ms=fit.params[term], hac_se=fit.bse[term],
                                     p=fit.pvalues[term], note="descriptive; panel-HAC (maxlags 5); trial-level pooled-mean differences"))
        for col in ("set_size", "correct", "artifact"):
            for lv, g in dfc.groupby(col):
                cov_rows.append(dict(comparison=name, term=f"group_mean[{col}={lv}]", coef_ms=g.d.mean(),
                                     hac_se=g.d.std(ddof=1) / np.sqrt(len(g)), p=np.nan, n_trials=len(g),
                                     note="group mean of trial-level difference (SE ignores dependence)"))
    diag = pd.DataFrame(diag_rows)
    diag.to_csv(TAB / "diagnostics_block_bootstrap_applicability.csv", index=False)
    pd.DataFrame(acf_rows).to_csv(TAB / "diagnostics_acf_trial_level_differences.csv", index=False)
    pd.DataFrame(trend_rows).to_csv(TAB / "diagnostics_within_session_trends.csv", index=False)
    pd.DataFrame(cov_rows).to_csv(TAB / "diagnostics_task_covariates.csv", index=False)
    # region-level ACF and series source data
    reg_acf = [dict(region=r, lag=l, acf=v) for j, r in enumerate(sc.REGIONS)
               for l, v in enumerate(acf_pooled(Mtr[:, j], sess_num)) if l > 0]
    pd.DataFrame(reg_acf).to_csv(TAB / "diagnostics_acf_trial_level_region_means.csv", index=False)
    series = meta[["session", "trial_pos", "trial_id", "artifact", "set_size", "correct"]].copy()
    for j, r in enumerate(sc.REGIONS):
        series[f"mean_{sc.REGION_SHORT[r]}"] = Mtr[:, j]
    for kk, (a, b) in enumerate(sc.COMPARISONS):
        series[f"diff_{sc.REGION_SHORT[sc.REGIONS[a]]}-{sc.REGION_SHORT[sc.REGIONS[b]]}"] = D6[:, kk]
    series.to_csv(FIG / "source_data/fig5_trial_series.csv", index=False)
    for r in diag.itertuples():
        REP.line(f"Diagnostic {r.comparison}: ACF1={r.acf_lag1:+.2f}, SE(L10)/SE(L5)={r.se_ratio_L10_L5:.2f}: {r.inference_condition_flags}")
    note_map = {(r.region_A, r.region_B): r.inference_condition_flags for r in diag.itertuples()}
    res["diagnostic_note"] = [note_map[(a, b)] for a, b in zip(res.region_A, res.region_B)]
    res.to_csv(TAB / "bootstrap_results_all_settings.csv", index=False)
    prim = res[res.is_primary]
    prim.to_csv(TAB / "primary_result_pooled_L5.csv", index=False)
    REP.line("Primary analysis (prespecified reference: pooled nonzero windows, all trials, L=5, B=50000):")
    for r in prim.itertuples():
        REP.line(f"  {sc.REGION_SHORT[r.region_A]}-{sc.REGION_SHORT[r.region_B]}: Δ={r.difference_ms:+.4f} ms | 95%CI "
                 f"[{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] | Bonferroni interval [{r.ci_bonf_low:+.4f}, {r.ci_bonf_high:+.4f}] | "
                 f"p_raw={r.raw_p:.2e} p_adj={r.adjusted_p:.2e} {r.significance_bonferroni}")
    save_state(bootstrap_streams=len(settings), bootstrap_variant_settings=n_variant_settings,
               refine_settings=[list(k) for k in refine_keys], bootstrap_rows=len(res))


# ============================================================================ S7, S8
def stage_s7():
    import figures
    figures.make_all(OUT, REP)


EXPECTED_OUTPUTS = [
    "tables/provenance_source_hashes.csv", "tables/provenance_nwb_sha256.csv", "tables/electrode_mapping_all_32_channels.csv",
    "tables/channel_pair_mapping.csv", "tables/trial_metadata.csv", "tables/old_checkpoint_verification.csv",
    "tables/checkpoint_reestimation_consistency.csv", "tables/trial_region_summary.csv", "tables/trial_pair_summary.csv",
    "tables/check_summary_reproduces_pooled_means.csv", "tables/window_level_descriptives_by_scope.csv",
    "tables/window_level_welch_audit_only.csv", "tables/historical_reference_check.csv",
    "tables/bootstrap_results_all_settings.csv", "tables/primary_result_pooled_L5.csv",
    "tables/diagnostics_block_bootstrap_applicability.csv", "tables/diagnostics_acf_trial_level_differences.csv",
    "tables/diagnostics_within_session_trends.csv", "tables/diagnostics_task_covariates.csv",
    "tables/welch_vs_new_test_comparison.csv", "tables/sensitivity_summary_wide.csv",
    "synthetic_validation/synthetic_validation_results.csv",
    "figures/fig1_fig4c_style_distribution_new_test.png", "figures/fig1_fig4c_style_distribution_new_test.pdf",
    "figures/fig2_primary_differences_ci.png", "figures/fig2_primary_differences_ci.pdf",
    "figures/fig3_session_specific_and_leave_one_out.png", "figures/fig3_session_specific_and_leave_one_out.pdf",
    "figures/fig4_sensitivity_block_length_zero_artifact_weights.png", "figures/fig4_sensitivity_block_length_zero_artifact_weights.pdf",
    "figures/fig5_trial_series_and_acf.png", "figures/fig5_trial_series_and_acf.pdf",
    "figures/fig5b_task_covariates.png", "figures/fig5b_task_covariates.pdf",
    "figures/fig6_welch_vs_new_table.png", "figures/fig6_welch_vs_new_table.pdf",
    "results_numeric_summary.md",
    "tables/session_heterogeneity_descriptive.csv", "tables/workspace_diff_vs_NeuroTD-main_zip.txt",
    "tables/posthoc_pair_level_heterogeneity.csv", "tables/posthoc_leave_one_electrode_out_summary.csv",
    "tables/posthoc_hemisphere_label_category.csv", "tables/posthoc_search_boundary_windows_by_region.csv",
    "figures/fig7_posthoc_electrode_composition.png", "figures/fig7_posthoc_electrode_composition.pdf",
    "analysis_plan_prespecified.md", "deviations_from_plan.md", "logs/time_estimate_at_start.json",
    "logs/environment.json", "run.log", "progress.jsonl", "timing_summary.csv",
]
NARRATIVE = ["REPORT.md", "methods_paragraph_EN.md", "response_reviewer_2_4_2_EN.md"]


def stage_s8():
    missing, empty = [], []
    for rel in EXPECTED_OUTPUTS + NARRATIVE:
        p = OUT / rel
        if not p.exists():
            missing.append(rel)
        elif p.stat().st_size == 0:
            empty.append(rel)
    res = pd.read_csv(TAB / "bootstrap_results_all_settings.csv")
    st = load_state()
    n_expected_rows = (len(LS) * len(VARIANTS_POOLED) + 6 * len(LS) * len(VARIANTS_SUB)) * 6
    n_main = int((res.run_tag == "main").sum())
    prim = res[res.is_primary]
    probs = []
    if n_main != n_expected_rows:
        probs.append(f"Bootstrap result rows {n_main}, expected {n_expected_rows}")
    if len(prim) != 6 or prim[["difference_ms", "raw_p", "ci95_low"]].isna().any().any():
        probs.append("Some of the six primary comparisons are missing")
    if (res.raw_p == 0).any():
        probs.append("Found p=0")
    syn = pd.read_csv(SYN / "synthetic_validation_results.csv")
    if not syn.passed.all():
        probs.append("Some synthetic checks failed")
    REP.line(f"Expected {len(EXPECTED_OUTPUTS) + len(NARRATIVE)} outputs: {len(missing)} missing, {len(empty)} empty; "
             f"bootstrap primary rows {n_main}/{n_expected_rows}; primary contrasts {len(prim)}; synthetic checks {int(syn.passed.sum())}/{len(syn)}")
    for m in missing:
        REP.line(f"  Missing: {m}")
    for p in probs:
        REP.line(f"  Issue: {p}")
    save_state(integrity_missing=missing, integrity_empty=empty, integrity_problems=probs)
    return not missing and not empty and not probs


def final_summary(complete: bool, t_run: float, failed_stage: str | None = None, remaining=()):
    st = load_state()
    if failed_stage:
        st.setdefault("integrity_problems", [])
        st["integrity_problems"] = [f"Failed stage: {failed_stage}"] + [f"Skipped stage: {s}" for s in remaining]
    REP.line("=" * 78)
    tim = pd.read_csv(OUT / "timing_summary.csv") if (OUT / "timing_summary.csv").exists() else pd.DataFrame()
    REP.line(f"Elapsed time for this invocation {fmt_hms(t_run)}")
    if len(tim):
        last = tim.groupby("stage").tail(1)
        for r in last.itertuples():
            REP.line(f"  {r.stage}: {fmt_hms(float(r.seconds))} ({r.status}, ended {r.end})")
        REP.line(f"  Total time across the latest runs of individual stages {fmt_hms(last.seconds.astype(float).sum())}")
    own, kids = peak_rss_mb()
    REP.line(f"Peak memory: parent {own:.0f} MB; children {kids:.0f} MB")
    REP.line(f"Reused trial/pair records {st.get('pairs_reused')}; new delay estimates for the primary analysis {st.get('pairs_recomputed', 0)}; "
             f"re-estimated {st.get('reestimate_computed_now')} pairs ({st.get('reestimate_identical')}/{st.get('reestimate_tasks')} matches); "
             f"failed {int(st.get('pairs_recompute_failed', 0) or 0) + int(st.get('reestimate_failed', 0) or 0)}")
    REP.line(f"Output directory: {OUT}")
    REP.line(f"Main report: {OUT / 'REPORT.md'}; numeric summary: {OUT / 'results_numeric_summary.md'}")
    REP.line(f"Principal figures: {FIG / 'fig1_fig4c_style_distribution_new_test.png'}, {FIG / 'fig2_primary_differences_ci.png'}, "
             f"{FIG / 'fig4_sensitivity_block_length_zero_artifact_weights.png'}, {FIG / 'fig5_trial_series_and_acf.png'}")
    if complete:
        REP.line("All stages complete")
    else:
        REP.line("Partially complete. Outstanding items: ")
        for m in st.get("integrity_missing", []) + st.get("integrity_problems", []):
            REP.line(f"  - {m}")


def main():
    global REP
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(k for k, _ in STAGES))
    args = ap.parse_args()
    want = args.stages.split(",")
    REP = Reporter(OUT)
    t_run = time.time()
    REP.line(f"Reviewer 2.4.2 / Figure 4c pipeline | PID {os.getpid()} | stages {want} | output {OUT}")
    REP.event("run_start", stages=want, pid=os.getpid())
    funcs = {"s1": stage_s1, "s2": stage_s2, "s3": stage_s3, "s4": stage_s4, "s5": stage_s5,
             "s6": stage_s6, "s7": stage_s7, "s8": stage_s8}
    complete = None
    for key, name in STAGES:
        if key not in want:
            continue
        REP.start_stage(name)
        try:
            out = funcs[key]()
            if key == "s8":
                complete = bool(out)
                REP.end_stage("complete" if complete else "complete with outstanding items")
            else:
                REP.end_stage("complete")
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            REP.line(f"[{name}] Error: {type(e).__name__}: {e}\n{tb}\nStopping this stage and all stages that depend on its output.")
            REP.event("error", error=str(e), traceback=tb)
            REP.end_stage("failed", note=str(e)[:200])
            keys = [k for k, _ in STAGES]
            remaining = [n for k, n in STAGES if k in want and keys.index(k) > keys.index(key)]
            final_summary(False, time.time() - t_run, failed_stage=name, remaining=remaining)
            REP.close()
            sys.exit(1)
    if complete is not None:
        final_summary(complete, time.time() - t_run)
    REP.event("run_end", seconds=round(time.time() - t_run, 1))
    REP.close()


if __name__ == "__main__":
    main()
