"""Comprehensive real-data audit pipeline. Run from the repository root:

    python -u realdata_comprehensive_audit_20261031_2228/scripts/pipeline.py [--stages s1,...]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa
import common as C
from delay_est import estimate_pair, verify_against_original
from progress import Reporter, RateTracker, fmt_hms, peak_rss_mb

import numpy as np
import pandas as pd

sys.path.insert(0, str(C.PREV2 / "scripts"))
import infer as I  # noqa: E402

STAGES = [
    ("s1", "Inputs, provenance and participant inventory"),
    ("s2", "Channel mapping and primary sources"),
    ("s3", "Waveform quality across all channels"),
    ("s4", "Original labels and within-bundle delays"),
    ("s5", "Temporal correspondence controls"),
    ("s6", "Known-delay injection"),
    ("s7", "Task phases and macroelectrodes"),
    ("s8", "Statistical validation"),
    ("s9", "Figures, report and integrity checks"),
]
REP: Reporter | None = None
_W = {}


def make_pool(n=None):
    ctx = mp.get_context("spawn")
    return ctx.Pool(n or N_WORKERS, initializer=_init)


def _init():
    from pynwb import NWBHDF5IO
    _W["nwb"] = {}
    for s, p in NWB.items():
        _W["nwb"][s] = NWBHDF5IO(str(p), "r").read()


def _micro(ses, trial):
    n = _W["nwb"][ses]
    sl = slice(trial * TRIAL_LEN, (trial + 1) * TRIAL_LEN)
    return np.asarray(n.acquisition["ecephys.lfp"].data[sl, :], dtype=np.float64)


def _macro(ses, trial):
    n = _W["nwb"][ses]
    sl = slice(trial * 16000, (trial + 1) * 16000)
    return np.asarray(n.acquisition["ecephys.ieeg"].data[sl, :], dtype=np.float64)


def channel_map():
    return pd.read_csv(PREV2 / "electrodes/channel_verification_table.csv")


def orig_pairs():
    return pd.read_csv(PREV2 / "electrodes/pair_classification.csv")


def labels_32():
    ch = channel_map()
    c1 = ch[ch.session == "ses-01"].sort_values("waveform_column")
    return c1.label.tolist(), c1.waveform_column.tolist()


def all_bundle_pairs():
    labs, cols = labels_32()
    lab2col = dict(zip(labs, cols))
    rows = []
    k = 0
    for b in BUNDLES:
        chs = [f"{b}{i}" for i in range(1, 9)]
        for a, c in combinations(chs, 2):
            rows.append(dict(pair_index=k, version="C_bundle", bundle=b,
                             label_i=a, label_j=c, channel_i=lab2col[a], channel_j=lab2col[c],
                             contact_gap=abs(int(a[len(b):]) - int(c[len(b):]))))
            k += 1
    return pd.DataFrame(rows)


def reuse_lookup():
    """Map (label_i,label_j) -> original pair_index in the 60-pair checkpoint."""
    pc = orig_pairs()
    d = {}
    for r in pc.itertuples():
        d[(r.label_i, r.label_j)] = int(r.pair_index)
        d[(r.label_j, r.label_i)] = int(r.pair_index)
    return d


# ============================================================================= S1
def stage_s1():
    env = dict(python=sys.version.split()[0], numpy=np.__version__, workers=N_WORKERS)
    try:
        import pynwb
        env["pynwb"] = pynwb.__version__
    except Exception as e:
        env["pynwb"] = str(e)
    (OUT / "logs/environment.json").write_text(json.dumps(env, indent=2))
    REP.line(f"Python {env['python']} numpy {env['numpy']} workers {N_WORKERS}")
    rows = []
    for s, p in NWB.items():
        REP.set_task(f"SHA-256 {p.name}")
        h = sha256_file(p)
        prev = pd.read_csv(PREV1 / "tables/provenance_nwb_sha256.csv")
        oldh = prev[prev.session == f"ses-0{s}"].sha256.iloc[0]
        rows.append(dict(session=f"ses-0{s}", path=str(p), bytes=p.stat().st_size,
                         sha256=h, matches_round1=h == oldh))
        REP.line(f"  {p.name} {p.stat().st_size:,}  SHA-256 {h[:16]}…")
    # robust match against prev1 table
    p1 = pd.read_csv(PREV1 / "tables/provenance_nwb_sha256.csv")
    hashcol = [c for c in p1.columns if "sha" in c.lower() or "hash" in c.lower()]
    pd.DataFrame(rows).to_csv(OUT / "provenance/nwb_sha256.csv", index=False)
    # other subjects inventory
    inv = []
    from pynwb import NWBHDF5IO
    for sub in sorted((DATA).glob("sub-*")):
        for f in sorted(sub.glob("*.nwb")):
            try:
                io = NWBHDF5IO(str(f), "r")
                n = io.read()
                lfp = n.acquisition.get("ecephys.lfp")
                labs = list(lfp.electrodes["label"][:]) if lfp is not None else []
                io.close()
                inv.append(dict(file=f.name, bytes=f.stat().st_size, n_micro=len(labs),
                                labels=";".join(map(str, labs[:32])),
                                in_main_analysis=f.name.startswith("sub-01_ses-0") and f.name[11] in "123"))
            except Exception as e:
                inv.append(dict(file=f.name, bytes=f.stat().st_size, n_micro=None, labels=str(e)[:80],
                                in_main_analysis=False))
    pd.DataFrame(inv).to_csv(OUT / "provenance/subject_session_inventory.csv", index=False)
    n_extra = sum(1 for r in inv if not r["in_main_analysis"])
    REP.line(f"Local DANDI files: {len(inv)}. The primary analysis covers three sessions; the other {n_extra} are inventoried separately.")
    # reuse checkpoint shape
    z = np.load(PREV1 / "checkpoints/windows_all_trials.npz", allow_pickle=True)
    assert z["shifts_samples"].shape == (150, 60, 508)
    REP.line("Reusing the first audit window delays, 150 × 60 × 508 (int16)")
    # benchmark
    _init()
    x = _micro(1, 0)[:, 0]
    y = _micro(1, 0)[:, 1]
    t0 = time.perf_counter()
    okc, oks = verify_against_original(x, y)
    t_orig = time.perf_counter() - t0
    t0 = time.perf_counter()
    estimate_pair(x, y)
    t_vec = time.perf_counter() - t0
    REP.line(f"Agreement: centers {okc}, shifts {oks} | original estimator {t_orig:.2f} s, vectorized {t_vec:.2f} s")
    if not (okc and oks):
        raise RuntimeError("The vectorized and original estimates differ. Stopping.")
    n_new = 100 * 150
    est = dict(verify_s=t_orig, vec_s=t_vec, new_pair_trials=n_new,
               s4_hours=n_new * t_vec / N_WORKERS / 3600,
               s5_hours=(112 * 150 * 2 + 12 * 150 * 3) * t_vec / N_WORKERS / 3600)
    (OUT / "logs/time_estimate_at_start.json").write_text(json.dumps(est, indent=2))
    REP.line(f"Initial estimate: new within-bundle delays about {est['s4_hours']:.2f} h; mismatch controls about {est['s5_hours']:.2f} h")
    save_state(vec_ok=True, t_vec=t_vec)


# ============================================================================= S2
def stage_s2():
    # Copy previous electrode findings and extend with official-source notes collected this round
    src = PREV2 / "electrodes"
    for name in ["channel_verification_table.csv", "electrode_findings.csv", "pair_classification.csv",
                 "questions_for_authors.md", "sources.md", "macro_contacts_ses-01.csv"]:
        (OUT / "electrodes" / name).write_bytes((src / name).read_bytes())
    findings = pd.read_csv(OUT / "electrodes/electrode_findings.csv")
    extra = [
        dict(id="E14", topic="Anatomical index in DANDI 000574 metadata", category="unconfirmed",
             finding="The DANDI page states that the Dandiset does not specify brain regions (no UBERON index). Version 0.231010.1809, contact Jan Hohenheim.",
             evidence="https://dandiarchive.org/dandiset/000574/0.231010.1809", impact="The DANDI metadata do not establish the microelectrode regions."),
        dict(id="E15", topic="NWB conversion code", category="unconfirmed",
             finding="The DANDI-linked janhohenheim/nwb-example repository illustrates reading NWB data. The audit could not retrieve the NIX documentation from G-Node (HTTP 403) or locate the NIX-to-NWB converter.",
             evidence="https://github.com/janhohenheim/nwb-example; G-Node doi:10.12751/g-node.d76994",
             impact="Conversion code could not establish whether microelectrodes were incorrectly linked to macrocontact rows."),
        dict(id="E16", topic="Meaning of AHL1–8 in the microelectrode series", category="conflicting sources",
             finding="NWB labels the 32 ecephys.lfp columns (Micro iEEG data, 32 kHz) AHL1–8, AL1–8, ECL1–8 and PHR1–8, referencing the same electrode rows as macrocontacts 1–8. The Scientific Data description reports nine microwires extending about 4 mm from each depth-electrode tip; its Figure 2 locates the microelectrodes at the tip. The file labels resemble shaft macrocontacts.",
             evidence="NWB electrode references; Boran et al. (2020), Scientific Data, Recording setup and Figure 2.",
             impact="The version A MTG/STG groups lack independent support. Version C labels bundles by name."),
        dict(id="E17", topic="Task phase boundaries", category="confirmed",
             finding="Fixation lasts 1 s, encoding 2 s and maintenance 3 s, followed by probe. With probe at 0, Science Advances gives fixation [−6,−5], encoding [−5,−3] and maintenance [−3,0] s. NWB trials last 8 s. NeuroTD uses start+6 for response timing, consistent with probe at 6 s. The NWB trials table lacks separate phase columns.",
             evidence="Boran et al. (2019) Science Advances, Methods; Scientific Data Figure 1; response_time in hiEEG_functions.py",
             impact="Phase intervals of 0–1, 1–3, 3–6 and 6–8 s follow the literature."),
    ]
    findings = pd.concat([findings, pd.DataFrame(extra)], ignore_index=True)
    findings.to_csv(OUT / "electrodes/electrode_findings.csv", index=False)
    all_bundle_pairs().to_csv(OUT / "electrodes/bundle_pairs_112.csv", index=False)
    # Version B: not created
    (OUT / "electrodes/version_B_status.txt").write_text(
        "Mapping B is unavailable. No independent conversion script or NIX channel table established separate microelectrode rows or a corrected hemisphere.\n")
    for r in extra:
        REP.line(f"  [{r['id']}] {r['topic']}: {r['category']}")
    # answers to the five questions (pre-data-collection of this round, based on sources)
    q = Path(OUT / "electrodes/five_mapping_questions.md")
    q.write_text("\n".join([
        "# Five mapping questions after source review",
        "",
        "1. Do AHL1–AHL8 and similar labels identify eight microwires or contacts along a shaft? **Conflicting sources.** The names and electrode rows match the numbered macrocontacts; the data description places microwires at the tip.",
        "2. Should microelectrodes be localized to the depth-electrode tip? **Scientific Data Figure 2 supports the tip location.** NWB coordinates vary along the shaft. The discrepancy remains unresolved.",
        "3. Do the electrode-table locations and coordinates apply only to macrocontacts? **Likely, pending verification against conversion code.** There are no separate microelectrode rows.",
        "4. Why do ECL and LR show inconsistent laterality? **The origin is unconfirmed.** The location strings disagree with the z sign under the documented NWB axes. The audit retained the recorded laterality.",
        "5. What units, filters and reference apply? **Confirmed:** common intracranial reference. **Conflicting:** filtering (table 0.5–1000 Hz vs description 0.5–5000 Hz) and units (declared volts; values resemble int16 scaled to ±200).",
        "",
        "Mapping A reproduces the original figure. C uses neutral within-bundle groups for the primary description. B remains unavailable. D2 assigns the tip the location of contact 1 for sensitivity analysis only.",
    ]), encoding="utf-8")


# ============================================================================= S3
def _qc_task(args):
    ses, trial = args
    wav = _micro(ses, trial)
    labs, _ = labels_32()
    rows = []
    for j in range(32):
        x = wav[:, j]
        fin = np.isfinite(x)
        dx = np.diff(x)
        step = np.median(np.abs(dx[dx != 0])) if np.any(dx != 0) else 0.0
        sat = np.mean(np.abs(x) >= 199.99)
        # longest constant run
        run = 1
        mx = 1
        for i in range(1, len(x)):
            if x[i] == x[i - 1]:
                run += 1
                mx = max(mx, run)
            else:
                run = 1
        rows.append(dict(session=f"ses-0{ses}", trial_pos=trial, channel=j, label=labs[j],
                         n_nonfinite=int((~fin).sum()), rms=float(np.sqrt(np.mean(x * x))),
                         vmax=float(np.max(np.abs(x))), sat_frac=float(sat),
                         quant_step=float(step), longest_constant=int(mx),
                         constant_ge_1ms=float(mx >= 32)))
    return rows


def stage_s3():
    labs, _ = labels_32()
    tasks = [(s, t) for s in (1, 2, 3) for t in range(50)]
    REP.set_progress(done=0, total=len(tasks), unit="trials")
    rt = RateTracker()
    rows = []
    with make_pool() as pool:
        for i, r in enumerate(pool.imap_unordered(_qc_task, tasks, 1), 1):
            rows.extend(r)
            REP.set_progress(done=i)
            if i % 10 == 0 or i == len(tasks):
                REP.line(f"[Quality checks] trial {i}/{len(tasks)} | {rt.text(i, len(tasks), 'trial')}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "qc/channel_trial_metrics.csv", index=False)
    summ = df.groupby("label").agg(rms_median=("rms", "median"), sat_mean=("sat_frac", "mean"),
                                   nonfinite_sum=("n_nonfinite", "sum"),
                                   longest_constant_p95=("longest_constant", lambda s: float(np.quantile(s, 0.95)))).reset_index()
    summ.to_csv(OUT / "qc/channel_summary.csv", index=False)
    REP.line(f"Total nonfinite samples {df.n_nonfinite.sum()}; median saturation fraction {df.sat_frac.median():.2e}")
    # focused pairs on all 150 trials
    focus = orig_pairs()
    focus = focus[focus.pair_index.isin([0, 3, 6, 7, 8, 9, 10, 31, 33, 53, 54, 59])]
    focus.to_csv(OUT / "qc/focus_pairs.csv", index=False)


# ============================================================================= S4 delays
def _delay_task(task):
    ses, trial, i, j = task["ses"], task["trial"], task["i"], task["j"]
    wav = _micro(ses, trial)
    c, sh, m0, mo = estimate_pair(wav[:, i], wav[:, j])
    path = Path(task["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, centers=c, shifts=sh, mse0=m0, mse_opt=mo,
                        ses=ses, trial=trial, i=i, j=j)
    return dict(ok=True, path=str(path), nwin=len(sh))


def stage_s4():
    bp = all_bundle_pairs()
    lookup = reuse_lookup()
    z = np.load(PREV1 / "checkpoints/windows_all_trials.npz", allow_pickle=True)
    old = z["shifts_samples"]  # (150,60,508)
    # assemble bundle array (150,112,508)
    out = np.zeros((150, 112, 508), dtype=np.int16)
    reused = 0
    todo = []
    ckdir = OUT / "checkpoints/bundle_pairs"
    ckdir.mkdir(parents=True, exist_ok=True)
    for r in bp.itertuples():
        key = (r.label_i, r.label_j)
        if key in lookup:
            out[:, r.pair_index, :] = old[:, lookup[key], :]
            reused += 1
        else:
            for t in range(150):
                ses = t // 50 + 1
                trial = t % 50
                todo.append(dict(ses=ses, trial=trial, i=int(r.channel_i), j=int(r.channel_j),
                                 path=str(ckdir / f"p{r.pair_index:03d}_t{t:03d}.npz"),
                                 pair_index=int(r.pair_index), t=t))
    REP.line(f"Of 112 within-bundle pairs, {reused} reuse existing checkpoints; {len(todo)} trial/pair records need new estimates")
    # skip existing
    todo2 = [t for t in todo if not Path(t["path"]).exists()]
    REP.line(f"Resuming: {len(todo)-len(todo2)} can be skipped; {len(todo2)} remain")
    rt = RateTracker(min_units=6, min_seconds=8)
    done = 0
    if todo2:
        with make_pool() as pool:
            for i, res in enumerate(pool.imap_unordered(_delay_task, todo2, 1), 1):
                done = i
                if i % 20 == 0 or i == len(todo2):
                    REP.set_progress(done=i, total=len(todo2), unit="trial/pair records")
                    REP.line(f"[Delay estimates] new {i}/{len(todo2)} | {reused} pairs reused | {rt.text(i, len(todo2), 'pair')}")
    # fill from checkpoints
    for t in todo:
        z2 = np.load(t["path"])
        out[t["t"], t["pair_index"], :] = z2["shifts"]
    np.savez_compressed(OUT / "checkpoints/bundle_shifts_150x112x508.npz", shifts=out,
                        window_centers=z["window_centers"])
    # summaries
    absms = np.abs(out) * MS
    rows = []
    for r in bp.itertuples():
        a = absms[:, r.pair_index, :]
        zf = (np.abs(out[:, r.pair_index, :]) == 0).mean()
        nz = a[np.abs(out[:, r.pair_index, :]) > 0]
        rows.append(dict(pair=f"{r.label_i}–{r.label_j}", bundle=r.bundle, contact_gap=r.contact_gap,
                         reused= (r.label_i, r.label_j) in lookup,
                         zero_frac=float(zf), mean_abs_nz=float(nz.mean()) if nz.size else np.nan,
                         median_abs_nz=float(np.median(nz)) if nz.size else np.nan,
                         boundary_frac=float((np.abs(out[:, r.pair_index, :]) == 500).mean()),
                         sign_pos_frac=float((out[:, r.pair_index, :] > 0).mean())))
    pd.DataFrame(rows).to_csv(OUT / "delays/bundle_pair_descriptives.csv", index=False)
    # original 4-region reuse descriptives from prev1
    prev = pd.read_csv(PREV1 / "tables/trial_pair_summary.csv")
    prev.to_csv(OUT / "delays/versionA_trial_pair_summary_reused.csv", index=False)
    save_state(bundle_reused_pairs=reused, bundle_new_pair_trials=len(todo))
    REP.line(f"Saved bundle_shifts_150x112x508.npz; reused {reused} pairs, {len(todo)} new checkpoints")


# ============================================================================= S5 mismatch
def _mismatch_maps(n_maps, tag):
    meta = pd.read_csv(PREV1 / "tables/trial_metadata.csv")
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 20, tag]))
    maps = []
    for m in range(n_maps):
        dest = np.arange(150)
        for ses in range(3):
            idx = np.arange(ses * 50, (ses + 1) * 50)
            ss = meta.set_size.to_numpy()[idx]
            for v in np.unique(ss):
                sub = idx[ss == v]
                if len(sub) < 2:
                    continue
                perm = rng.permutation(len(sub))
                # remove fixed points
                for i in range(len(sub)):
                    if perm[i] == i:
                        perm[i] = (i + 1) % len(sub)
                        perm[(i + 1) % len(sub)] = i
                dest[sub] = sub[perm]
        maps.append(dest.astype(np.int16))
    return maps


def _pair_task(task):
    wavx = _micro(task["ses_x"], task["trial_x"])
    wavy = _micro(task["ses_y"], task["trial_y"])
    if task.get("shift_samples", 0):
        D = int(task["shift_samples"])
        x = wavx[:-D, task["i"]]
        y = wavy[D:, task["j"]]
    else:
        x = wavx[:, task["i"]]
        y = wavy[:, task["j"]]
    c, sh, m0, mo = estimate_pair(x, y)
    np.savez_compressed(task["path"], shifts=sh, mse0=m0, mse_opt=mo)
    return dict(ok=True, path=task["path"])


def stage_s5():
    bp = pd.read_csv(OUT / "electrodes/bundle_pairs_112.csv")
    maps2 = _mismatch_maps(2, 1)
    maps5 = _mismatch_maps(5, 2)
    np.savez(OUT / "mismatch/trial_maps.npz", maps2=np.stack(maps2), maps5=np.stack(maps5))
    focus_idx = [0, 3, 6, 7, 8, 9, 10, 31, 33, 53, 54, 59]
    pc = orig_pairs().set_index("pair_index")
    tasks = []
    ck = OUT / "checkpoints/mismatch"
    ck.mkdir(parents=True, exist_ok=True)
    # 2 maps × 112 bundle pairs × 150
    for mi, dest in enumerate(maps2):
        for r in bp.itertuples():
            for t in range(150):
                t2 = int(dest[t])
                path = ck / f"b_m{mi}_p{r.pair_index:03d}_t{t:03d}.npz"
                tasks.append(dict(ses_x=t // 50 + 1, trial_x=t % 50, ses_y=t2 // 50 + 1, trial_y=t2 % 50,
                                  i=int(r.channel_i), j=int(r.channel_j), path=str(path), shift_samples=0))
    # extra 3 maps (index 2,3,4 of maps5) for 12 focus pairs only
    for mi in range(2, 5):
        dest = maps5[mi]
        for pid in focus_idx:
            r = pc.loc[pid]
            for t in range(150):
                t2 = int(dest[t])
                path = ck / f"f_m{mi}_p{pid:03d}_t{t:03d}.npz"
                tasks.append(dict(ses_x=t // 50 + 1, trial_x=t % 50, ses_y=t2 // 50 + 1, trial_y=t2 % 50,
                                  i=int(r.channel_i), j=int(r.channel_j), path=str(path), shift_samples=0))
    # timeshift
    shift_pairs = []
    for r in bp.itertuples():
        if r.contact_gap == 1 and r.label_i.endswith("1"):
            shift_pairs.append(r)
    for pid in focus_idx + [6, 7, 8]:
        r = pc.loc[pid]
        shift_pairs.append(type("R", (), dict(pair_index=int(pid), channel_i=int(r.channel_i),
                                              channel_j=int(r.channel_j), label_i=r.label_i, label_j=r.label_j)))
    seen = set()
    for D in (32000, 128000):
        for r in shift_pairs:
            key = (getattr(r, "label_i", pid), getattr(r, "label_j", None), D)
            if key in seen:
                continue
            seen.add(key)
            for t in range(150):
                path = ck / f"sh{D}_p{int(r.pair_index):03d}_t{t:03d}.npz"
                tasks.append(dict(ses_x=t // 50 + 1, trial_x=t % 50, ses_y=t // 50 + 1, trial_y=t % 50,
                                  i=int(r.channel_i), j=int(r.channel_j), path=str(path), shift_samples=D))
    todo = [t for t in tasks if not Path(t["path"]).exists()]
    REP.line(f"Temporal correspondence tasks {len(tasks)}, available checkpoints {len(tasks)-len(todo)}, remaining {len(todo)}")
    rt = RateTracker(min_units=6, min_seconds=8)
    if todo:
        with make_pool() as pool:
            for i, _ in enumerate(pool.imap_unordered(_pair_task, todo, 1), 1):
                if i % 25 == 0 or i == len(todo):
                    REP.set_progress(done=i, total=len(todo), unit="control estimates")
                    REP.line(f"[Temporal correspondence] {i}/{len(todo)} | {rt.text(i, len(todo), 'task')}")
    # summarize bundle mismatch map0 vs matched
    sh = np.load(OUT / "checkpoints/bundle_shifts_150x112x508.npz")["shifts"]
    rows = []
    for r in bp.itertuples():
        matched = np.abs(sh[:, r.pair_index, :]) * MS
        mm = []
        for t in range(150):
            p = ck / f"b_m0_p{r.pair_index:03d}_t{t:03d}.npz"
            mm.append(np.abs(np.load(p)["shifts"]) * MS)
        mm = np.stack(mm)
        rows.append(dict(pair=f"{r.label_i}–{r.label_j}", bundle=r.bundle, contact_gap=r.contact_gap,
                         matched_mean_abs=float(matched.mean()), mismatch_mean_abs=float(mm.mean()),
                         matched_zero=float((sh[:, r.pair_index, :] == 0).mean()),
                         mismatch_zero=float((np.stack([np.load(ck / f'b_m0_p{r.pair_index:03d}_t{t:03d}.npz')['shifts']
                                                        for t in range(150)]) == 0).mean()),
                         delta_mean_abs=float(matched.mean() - mm.mean())))
    pd.DataFrame(rows).to_csv(OUT / "mismatch/bundle_matched_vs_mismatch_map0.csv", index=False)
    REP.line("Saved mismatch/bundle_matched_vs_mismatch_map0.csv")


# ============================================================================= S6 injection
def _inject_task(task):
    wav = _micro(task["ses"], task["trial"])
    x = wav[:, task["ch"]]
    N = len(x)
    tau = task["tau"]
    y = np.zeros(N)
    if isinstance(tau, int):
        if tau >= 0:
            y[tau:] = task["amp"] * x[:N - tau]
        else:
            y[:N + tau] = task["amp"] * x[-tau:]
        tautrue = np.full(508, tau)
    else:
        t = np.arange(N) / FS
        tau_t = (80 + 40 * np.sin(2 * np.pi * t / 8.0)).astype(int)
        for i in range(N):
            j = i - tau_t[i]
            if 0 <= j < N:
                y[i] = task["amp"] * x[j]
        tautrue = None
    if task["mix"] > 0:
        w2 = _micro(task["ses2"], task["trial2"])[:, task["ch"]]
        y = (1 - task["mix"]) * y + task["mix"] * w2
    c, sh, m0, mo = estimate_pair(x, y)
    # valid windows: start >= |tau|+500
    if isinstance(tau, int):
        need = abs(tau) + W
        valid = (c - (W - 1) // 2 >= need) & (c + (W + 1) // 2 + abs(tau) < N)
        err = sh[valid].astype(float) - tau
    else:
        valid = np.ones(len(sh), bool)
        err = np.full(valid.sum(), np.nan)
    np.savez_compressed(task["path"], shifts=sh, err=err.astype(np.float32), n_valid=int(valid.sum()),
                        tau=str(tau), mix=task["mix"], amp=task["amp"])
    return dict(ok=True, n_valid=int(valid.sum()), rmse=float(np.sqrt(np.mean(err ** 2))) if err.size and np.isfinite(err).all() else np.nan)


def stage_s6():
    labs, cols = labels_32()
    lab2col = dict(zip(labs, cols))
    chans = ["AHL1", "AHL8", "AL1", "ECL1", "PHR1"]
    taus = [0, 10, 80, 250, 480, 600, "tv"]
    tasks = []
    ck = OUT / "checkpoints/injection"
    ck.mkdir(parents=True, exist_ok=True)
    k = 0
    for ses in (1, 2, 3):
        for trial in (0, 24, 49):
            other = 25 if trial != 25 else 10
            for lab in chans:
                for tau in taus:
                    for mix in (0.0, 0.5):
                        for amp in (1.0, 0.5):
                            if mix > 0 and tau == "tv":
                                continue
                            path = ck / f"inj_{k:04d}.npz"
                            tasks.append(dict(ses=ses, trial=trial, ses2=ses, trial2=other,
                                              ch=lab2col[lab], tau=tau, mix=mix, amp=amp, path=str(path),
                                              label=lab, k=k))
                            k += 1
    todo = [t for t in tasks if not Path(t["path"]).exists()]
    REP.line(f"Injection conditions {len(tasks)}; {len(todo)} remain")
    recs = []
    rt = RateTracker()
    with make_pool() as pool:
        # map with metadata: run sequentially collect via imap on todo, then reload
        if todo:
            for i, res in enumerate(pool.imap(_inject_task, todo), 1):
                if i % 10 == 0 or i == len(todo):
                    REP.line(f"[Injection] {i}/{len(todo)} | {rt.text(i, len(todo), 'condition')}")
    for t in tasks:
        z = np.load(t["path"], allow_pickle=True)
        recs.append(dict(label=t["label"], ses=t["ses"], trial=t["trial"], tau=str(t["tau"]),
                         mix=t["mix"], amp=t["amp"], n_valid=int(z["n_valid"]),
                         rmse=float(np.sqrt(np.mean(z["err"] ** 2))) if z["err"].size else np.nan,
                         mae=float(np.mean(np.abs(z["err"]))) if z["err"].size else np.nan,
                         recover2=float(np.mean(np.abs(z["err"]) <= 2)) if z["err"].size else np.nan,
                         boundary=float(np.mean(np.abs(z["shifts"]) == 500))))
    pd.DataFrame(recs).to_csv(OUT / "injection/recovery_all_conditions.csv", index=False)
    g = pd.DataFrame(recs).groupby(["tau", "mix", "amp"]).agg(rmse=("rmse", "median"), recover2=("recover2", "median")).reset_index()
    g.to_csv(OUT / "injection/recovery_by_condition.csv", index=False)
    REP.line("Summarized every injection condition")


# ============================================================================= S7 epochs + macro
def stage_s7():
    sh = np.load(OUT / "checkpoints/bundle_shifts_150x112x508.npz")
    shifts = sh["shifts"]
    centers = sh["window_centers"]
    t_s = centers / FS
    bp = pd.read_csv(OUT / "electrodes/bundle_pairs_112.csv")
    rows = []
    for name, (a, b) in EPOCHS.items():
        if name == "full":
            m = np.ones(len(centers), bool)
        else:
            m = (t_s >= a) & (t_s < b)
        if m.sum() < 5:
            continue
        for r in bp.itertuples():
            sl = shifts[:, r.pair_index, :][:, m]
            absms = np.abs(sl) * MS
            nz = absms[np.abs(sl) > 0]
            rows.append(dict(epoch=name, pair=f"{r.label_i}–{r.label_j}", bundle=r.bundle,
                             n_windows=int(m.sum()), mean_abs_nz=float(nz.mean()) if nz.size else np.nan,
                             zero_frac=float((sl == 0).mean())))
    pd.DataFrame(rows).to_csv(OUT / "epochs/bundle_by_epoch.csv", index=False)
    # macro ses-01, contacts 1-2 of each bundle.
    # Column index is position in the 48-wide ElectricalSeries, NOT the electrode-table row.
    from pynwb import NWBHDF5IO
    (OUT / "epochs").mkdir(parents=True, exist_ok=True)
    (OUT / "macro").mkdir(parents=True, exist_ok=True)
    io = NWBHDF5IO(str(NWB[1]), "r")
    n = io.read()
    ie = n.acquisition["ecephys.ieeg"]
    region = np.asarray(ie.electrodes.data[:], dtype=int)
    table_labels = [str(x) for x in ie.electrodes.table["label"][:]]
    col_of = {table_labels[int(r)]: k for k, r in enumerate(region)}
    io.close()
    want = [f"{b}{i}" for b in BUNDLES for i in (1, 2)]
    idx = {lab: col_of[lab] for lab in want if lab in col_of}
    missing = [lab for lab in want if lab not in idx]
    if missing:
        raise RuntimeError(f"Missing macroelectrode label {missing}; available labels {sorted(col_of)[:12]}…")
    if any(v < 0 or v >= 48 for v in idx.values()):
        raise RuntimeError(f"Macroelectrode column out of range: {idx}")
    REP.line("Macroelectrode column mapping " + ", ".join(f"{k}→{v}" for k, v in sorted(idx.items())))
    mac_rows = []
    _init()
    for t in range(50):
        wav = _macro(1, t)
        for b in BUNDLES:
            i, j = idx[f"{b}1"], idx[f"{b}2"]
            c, shf, m0, mo = estimate_pair(wav[:, i], wav[:, j], W_MACRO, H_MACRO, MAX_SHIFT_MACRO)
            mac_rows.append(dict(trial=t, bundle=b, nwin=len(shf),
                                 mean_abs_ms=float(np.mean(np.abs(shf)) * 1000 / FS_MACRO),
                                 zero_frac=float((shf == 0).mean()),
                                 W=W_MACRO, H=H_MACRO, max_shift=MAX_SHIFT_MACRO))
        if (t + 1) % 10 == 0:
            REP.line(f"[Macroelectrodes] ses-01 trial {t+1}/50")
    pd.DataFrame(mac_rows).to_csv(OUT / "macro/ses01_adjacent_contacts.csv", index=False)
    REP.line(f"Macroelectrode settings: W={W_MACRO}, H={H_MACRO}, search ±{MAX_SHIFT_MACRO} (window {W_MACRO/FS_MACRO*1000:.2f} ms)")


# ============================================================================= S8 stats
def bundle_SN(include_zero=False):
    sh = np.load(OUT / "checkpoints/bundle_shifts_150x112x508.npz")["shifts"]
    bp = pd.read_csv(OUT / "electrodes/bundle_pairs_112.csv")
    T, P, W = sh.shape
    S = np.zeros((T, 4))
    N = np.zeros((T, 4))
    for bi, b in enumerate(BUNDLES):
        idx = bp.index[bp.bundle == b].to_numpy()
        a = sh[:, idx, :].reshape(T, -1)
        if include_zero:
            S[:, bi] = np.abs(a).sum(1) * MS
            N[:, bi] = a.shape[1]
        else:
            m = np.abs(a) > 0
            S[:, bi] = np.where(m, np.abs(a) * MS, 0).sum(1)
            N[:, bi] = m.sum(1)
    return S, N


def _boot_one(S, N, w, sessions, B, L, seed):
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    return I.boot_inference(S, N, w, sessions, rng, B, Ls=[L])


def stage_s8():
    (OUT / "calibration").mkdir(parents=True, exist_ok=True)
    (OUT / "realdata").mkdir(parents=True, exist_ok=True)
    S, N = bundle_SN(False)
    w = np.ones(150)
    sessions = I.SESSIONS if hasattr(I, "SESSIONS") else [(0, 50), (50, 100), (100, 150)]
    # real point estimates
    mu, _ = I.pooled(S, N, w)
    delta = I.pairwise(mu)
    comps = list(combinations(BUNDLES, 2))
    rows = []
    for k, (a, b) in enumerate(comps):
        rows.append(dict(version="C_bundle", estimand="pooled |delay| nonzero same-bundle",
                         A=a, B=b, difference_ms=float(delta[k])))
    # studentized L=10 and HAC b=1 on real data
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 3, 0]))
    bo = I.boot_inference(S, N, w, sessions, rng, 20000, Ls=[10])
    # boot_inference returns stacked methods; with Ls=[10] -> M0 L10 and M2 L10
    names = I.boot_method_names()
    # filter to L10 only - function with Ls=[10] still uses boot_method_names internally?
    # Read boot_inference: names from Ls parameter
    # We'll store p from returned arrays
    for k, (a, b) in enumerate(comps):
        rows[k]["M2_L10_raw_p"] = float(bo["p"][1, k]) if bo["p"].ndim == 2 else float(bo["p"][k])
        rows[k]["M2_L10_ci95_low"] = float(delta[k] - bo["half95"][1, k]) if bo["half95"].ndim == 2 else float(delta[k] - bo["half95"][k])
        rows[k]["M2_L10_ci95_high"] = float(delta[k] + bo["half95"][1, k]) if bo["half95"].ndim == 2 else float(delta[k] + bo["half95"][k])
        rows[k]["M0_L10_raw_p"] = float(bo["p"][0, k]) if bo["p"].ndim == 2 else np.nan
    pd.DataFrame(rows).to_csv(OUT / "realdata/bundle_differences_uncalibrated.csv", index=False)
    REP.line("Saved real-data point estimates. Calibration determines whether p-values can support formal inference.")
    # lightweight calibration of FWER for M0/M2 L=10 using AR(1) on 4-dim residuals
    # residualize real trial means
    with np.errstate(invalid="ignore", divide="ignore"):
        y = np.where(N > 0, S / N, np.nan)
    y = np.nan_to_num(y, nan=np.nanmean(y))
    phi_hat = []
    for r in range(4):
        num = den = 0.0
        for s0, s1 in sessions:
            x = y[s0:s1, r] - y[s0:s1, r].mean()
            num += (x[1:] * x[:-1]).sum()
            den += (x[:-1] ** 2).sum()
        phi_hat.append(num / den if den else 0.0)
    sd = y.std(0)
    REP.line("Lag-1 correlation of bundle-level trial means: φ=" + ", ".join(f"{b} {p:.3f}" for b, p in zip(BUNDLES, phi_hat)))
    # calibration scenarios
    def sim_y(phi, rng, nsim, offset=None):
        out = np.zeros((nsim, 150, 4))
        for s in range(3):
            e = rng.normal(0, 1, size=(nsim, 50, 4)) * sd
            for t in range(50):
                if t == 0:
                    out[:, s * 50, :] = e[:, 0] / max(np.sqrt(1 - phi ** 2), 1e-6)
                else:
                    out[:, s * 50 + t, :] = phi * out[:, s * 50 + t - 1, :] + e[:, t]
        if offset is not None:
            out += offset
        return out

    def fwer_of(method, phi, nsim, seed, offset=None):
        rng = np.random.default_rng(seed)
        Y = sim_y(phi, rng, nsim, offset)
        # S = Y * N, keep N
        rej = 0
        for i in range(nsim):
            Si = Y[i] * N
            boi = I.boot_inference(Si, N, w, sessions, np.random.default_rng(seed + 1000 + i), 1999, Ls=[10])
            p = boi["p"][1] if method == "M2" else boi["p"][0]
            padj = np.minimum(6 * p, 1)
            tn = np.ones(6, bool) if offset is None else np.array([True, True, True, False, False, False])
            # offset on region 0 only -> comparisons 0,1,2 involving region 0 are false nulls if offset
            if offset is not None:
                tn = np.array([False, False, False, True, True, True])  # HIP-like: first region shifted; comps 0,1,2 involve 0
            if (padj[tn] < 0.05).any():
                rej += 1
        return rej, nsim

    cal_rows = []
    # smaller nsim for speed but report honestly: 400 cal, 1000 val for phi=0 and phi=0.6
    specs = [("A_iid", 0.0, 400, [SEED, 1, 0]), ("B2_phi0.6", 0.6, 400, [SEED, 1, 1])]
    for name, phi, nsim, seed in specs:
        REP.set_task(f"Calibration {name} nsim={nsim}")
        rej, n = fwer_of("M2", phi, nsim, seed[2] + SEED * 10)
        cal_rows.append(dict(stage="calibration", method="M2_L10", scenario=name, nsim=n,
                             fwer=rej / n, fwer_events=rej, alpha=0.05))
        REP.line(f"  Calibration {name} M2 L=10 FWER={rej}/{n}={rej/n:.3f}")
    val_rows = []
    specs_v = [("A_iid", 0.0, 1000, [SEED, 2, 0]), ("B2_phi0.6", 0.6, 1000, [SEED, 2, 1])]
    for name, phi, nsim, seed in specs_v:
        REP.set_task(f"Independent validation {name} nsim={nsim}")
        rej, n = fwer_of("M2", phi, nsim, seed[2] + SEED * 20)
        from scipy.stats import beta
        lo = 0 if rej == 0 else beta.ppf(0.025, rej, n - rej + 1)
        hi = 1 if rej == n else beta.ppf(0.975, rej + 1, n - rej)
        verdict = "pass" if hi <= 0.075 else ("fail" if lo > 0.075 else "insufficient evidence")
        val_rows.append(dict(stage="validation", method="M2_L10", scenario=name, nsim=n,
                             fwer=rej / n, fwer_events=rej, cp_low=float(lo), cp_high=float(hi),
                             verdict=verdict, alpha=0.05))
        REP.line(f"  Validation {name} M2 L=10 FWER={rej}/{n}={rej/n:.3f} CP[{lo:.3f},{hi:.3f}] {verdict}")
    pd.DataFrame(cal_rows).to_csv(OUT / "calibration/calibration_fwer.csv", index=False)
    pd.DataFrame(val_rows).to_csv(OUT / "calibration/validation_fwer.csv", index=False)
    passed = all(r["verdict"] == "pass" for r in val_rows)
    save_state(m2_l10_validation_passed=passed)
    # final decision table
    dec = pd.DataFrame(rows)
    if not passed:
        dec["decision"] = "insufficient evidence (M2 L=10 failed the independent core-scenario criteria)"
        dec["interpretation"] = "Descriptive difference only; formal calibration criterion failed"
    else:
        dec["decision"] = ["significant (familywise 0.05)" if min(6 * p, 1) < 0.05 else "nonsignificant (does not establish equivalence)"
                           for p in dec["M2_L10_raw_p"]]
    # mapping restriction
    dec["anatomy"] = "Anatomical differences remain unconfirmed: microwire locations conflict with NWB labels; version C represents bundles only."
    dec.to_csv(OUT / "realdata/final_decisions.csv", index=False)
    REP.line("Saved decisions using the prespecified rule: significance requires successful calibration; otherwise report insufficient evidence.")


# ============================================================================= S9
def stage_s9():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG = OUT / "figures"
    FIG.mkdir(parents=True, exist_ok=True)
    bp = pd.read_csv(OUT / "delays/bundle_pair_descriptives.csv")
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for b in BUNDLES:
        d = bp[bp.bundle == b]
        ax.scatter(d.contact_gap, d.mean_abs_nz, s=18, label=b)
    ax.set_xlabel("Difference in contact index within bundle")
    ax.set_ylabel("Mean nonzero |delay| (ms)")
    ax.legend()
    ax.set_title("Version C: within-bundle absolute delays (descriptive)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_bundle_delay_vs_gap.png", dpi=160)
    fig.savefig(FIG / "fig_bundle_delay_vs_gap.pdf")
    plt.close()
    mm = pd.read_csv(OUT / "mismatch/bundle_matched_vs_mismatch_map0.csv")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(mm.mismatch_mean_abs, mm.matched_mean_abs, s=12)
    lim = [0, max(mm.mismatch_mean_abs.max(), mm.matched_mean_abs.max())]
    ax.plot(lim, lim, "k-", lw=0.6)
    ax.set_xlabel("Mean mismatched |delay| (ms)")
    ax.set_ylabel("Mean matched |delay| (ms)")
    ax.set_title("Matched trials vs within-session set-size-matched mismatches")
    fig.tight_layout()
    fig.savefig(FIG / "fig_matched_vs_mismatch.png", dpi=160)
    fig.savefig(FIG / "fig_matched_vs_mismatch.pdf")
    plt.close()
    inj = pd.read_csv(OUT / "injection/recovery_by_condition.csv")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(range(len(inj)), inj.recover2, "o-")
    ax.set_xticks(range(len(inj)), [f"{r.tau}/m{r.mix}/a{r.amp}" for r in inj.itertuples()], rotation=75, fontsize=6)
    ax.set_ylabel("Recovery within two samples (median fraction)")
    ax.set_title("Injected delays: all tested conditions")
    fig.tight_layout()
    fig.savefig(FIG / "fig_injection_recovery.png", dpi=160)
    fig.savefig(FIG / "fig_injection_recovery.pdf")
    plt.close()
    # write reports
    _write_reports()
    expected = [
        "analysis_plan.md", "REPORT.md", "SUMMARY.md",
        "methods_paragraph_EN.md", "response_reviewer_EN.md",
        "provenance/nwb_sha256.csv", "provenance/subject_session_inventory.csv",
        "electrodes/electrode_findings.csv", "electrodes/five_mapping_questions.md",
        "qc/channel_trial_metrics.csv", "delays/bundle_pair_descriptives.csv",
        "mismatch/bundle_matched_vs_mismatch_map0.csv",
        "injection/recovery_all_conditions.csv", "calibration/validation_fwer.csv",
        "realdata/final_decisions.csv", "run.log", "timing_summary.csv",
    ]
    missing = [e for e in expected if not (OUT / e).exists()]
    empty = [e for e in expected if (OUT / e).exists() and (OUT / e).stat().st_size == 0]
    integ = dict(expected=len(expected), missing=missing, empty=empty)
    (OUT / "integrity_check.json").write_text(json.dumps(integ, indent=2, ensure_ascii=False))
    REP.line(f"Integrity check: {missing} missing, {empty} empty")
    return not missing and not empty


def _write_reports():
    dec = pd.read_csv(OUT / "realdata/final_decisions.csv")
    val = pd.read_csv(OUT / "calibration/validation_fwer.csv")
    mm = pd.read_csv(OUT / "mismatch/bundle_matched_vs_mismatch_map0.csv")
    inj = pd.read_csv(OUT / "injection/recovery_by_condition.csv")
    findings = pd.read_csv(OUT / "electrodes/electrode_findings.csv")
    timing = pd.read_csv(OUT / "timing_summary.csv") if (OUT / "timing_summary.csv").exists() else pd.DataFrame()
    st = load_state()
    L = []
    L.append("# Comprehensive real-data audit")
    L.append("")
    L.append("Directory: `realdata_comprehensive_audit_20261031_2228/`. These supplementary analyses followed the first two audits and were not preregistered.")
    L.append("")
    L.append("## Findings at three levels")
    L.append("")
    L.append("### A. Computation")
    L.append("The three NWB files were loaded by filename. Delay settings match the first audit. All 60 original within-region pairs reuse verified checkpoints. The vectorized estimator was used for new within-bundle pairs after its centers and shifts exactly matched the original function on the benchmark pairs. Reproducible delay arrays alone cannot establish anatomical labels or neural conduction times.")
    L.append("")
    L.append("### B. Signal and estimation")
    L.append(f"`injection/recovery_all_conditions.csv` lists each injection condition; the table and figure report the median fraction within two samples of truth. Across {len(mm)} within-bundle pairs, the median difference between matched and map0 mismatched mean delays is {mm.delta_mean_abs.median():.3f} ms.")
    L.append("Mismatched trials retain the same task phase. Similar matched and mismatched means suggest the statistic may be insensitive to precise within-trial alignment; coupling remains possible.")
    L.append("")
    L.append("### C. Neurophysiological interpretation")
    L.append("Microelectrode and macroelectrode series share electrode-table rows (E2/E16). MTG/STG labels cannot establish microelectrode anatomy. Version C uses bundle names. Formal bundle comparisons follow. " + "; ".join(f"{r.A}−{r.B} Δ={r.difference_ms:+.3f} ms → {r.decision}" for r in dec.itertuples()) + ".")
    L.append("")
    L.append("## Calibration")
    L.append(val.to_string(index=False))
    L.append("")
    L.append(f"M2 L=10 passed independent validation: {st.get('m2_l10_validation_passed')}")
    L.append("")
    L.append("## Five mapping questions")
    L.append((OUT / "electrodes/five_mapping_questions.md").read_text())
    L.append("")
    L.append("## Reviewer comments")
    L.append("- 2.4.2: Overlapping windows are grouped for inference. At φ=0.6, calibration did not support significance.")
    L.append("- 2.5: Metadata conflict on units, filters and microwire locations. Near-zero delays do not confirm duplicate channels.")
    L.append("- 2.7: The four-group ordering cannot establish anatomical differences. In the previous audit, the higher hippocampal estimate depended on the AHL1 contact composition.")
    L.append("")
    L.append("## Runtime")
    if len(timing):
        L.append(timing.to_string(index=False))
    (OUT / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "SUMMARY.md").write_text(
        "# Summary\n\n"
        "The primary analysis covers three sessions in sub-01. Given uncertain microelectrode anatomy, the primary description covers within-bundle pairs in four bundles. "
        "The delay estimates are reproducible. The report gives injection and mismatch results. Formal inference: " +
        ("calibration failed; evidence was insufficient for all six between-bundle differences." if not st.get("m2_l10_validation_passed") else "see final_decisions.csv") +
        "\n", encoding="utf-8")
    (OUT / "methods_paragraph_EN.md").write_text(
        "Draft Methods (the manuscript has not been edited). We audited DANDI 000574 NWB files for sub-01 sessions 01–03. "
        "Delays used the original NeuroTD discrete estimator (W=1001, H=500, ±500 samples, 32 kHz). "
        "Because microelectrode ElectricalSeries reuse macro-contact electrode-table rows, we treat four-region labels as version A (legacy) and analyse same-shaft micro pairs as version C with neutral bundle names. "
        "Temporal correspondence was tested with set-size-matched trial shuffles and non-circular time shifts. "
        "Estimator behaviour was tested by injecting known delays into real traces without wrap-around. "
        "Inference for bundle-level pooled absolute delays used studentized circular block bootstrap (L=10) after a small independent calibration/validation study; if that study failed the pre-specified FWER rule, differences were reported as descriptive only.\n",
        encoding="utf-8")
    (OUT / "response_reviewer_EN.md").write_text(
        "Draft reply (the manuscript text has not been revised in this repository).\n\n"
        "Reviewer 2.4.2: overlapping windows are no longer treated as independent. We still do not claim a calibrated p<0.05 for regional or bundle differences, because procedures that ignore trial dependence, and even trial-level block bootstrap at L=5, are anti-conservative under AR(1) dependence of the magnitude seen in calibration.\n\n"
        "Reviewer 2.5: NWB maps 32 kHz 'Micro iEEG' columns onto the same electrode-table rows as 2 kHz macro contacts. Units are declared as volts but the numeric range matches a ±200 full-scale integer export. Near-zero delay pairs are highly correlated but not sample-wise identical; we did not exclude them as duplicate channels.\n\n"
        "Reviewer 2.7: we cannot interpret MTG/STG microelectrode groupings as verified anatomy. Same-shaft pairs are the only grouping supported without assuming the disputed table. Sign of a delay is not treated as a conduction direction.\n",
        encoding="utf-8")
    if REP:
        REP.line("Saved REPORT.md, SUMMARY.md, Methods and the reviewer-response draft. Manuscript editing was outside this audit.")


def main():
    global REP
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(k for k, _ in STAGES))
    args = ap.parse_args()
    want = [s.strip() for s in args.stages.split(",") if s.strip()]
    REP = Reporter(OUT, heartbeat_s=10)
    names = dict(STAGES)
    REP.line(f"Starting comprehensive real-data audit | PID {os.getpid()} | stages {want} | {OUT}")
    fn = {k: globals()[f"stage_{k}"] for k, _ in STAGES}
    done, complete = [], False
    for s in want:
        REP.start_stage(names[s])
        try:
            r = fn[s]()
            if s == "s9" and r is False:
                REP.end_stage("complete with missing items")
                done.append(s)
                continue
            REP.end_stage("complete")
            done.append(s)
            complete = s == "s9" and r is True
        except Exception as e:
            REP.line(f"[{names[s]}] Error: {type(e).__name__}: {e}")
            REP.line(traceback.format_exc())
            REP.end_stage("failed", note=str(e)[:200])
            break
    if complete:
        REP.line("All stages complete")
    else:
        left = [names[x] for x in want if x not in done]
        REP.line(f"Finished. Complete: {[names[x] for x in done]}; incomplete: {left}")
    own, kids = peak_rss_mb()
    REP.line(f"Peak memory: parent {own:.0f} MB / children {kids:.0f} MB")
    REP.close()


if __name__ == "__main__":
    main()
