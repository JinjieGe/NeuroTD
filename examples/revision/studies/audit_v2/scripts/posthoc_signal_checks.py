"""Post-hoc (not pre-specified) signal checks prompted by the stage-3 waveform results.

A. Amplitude scale, int16 quantization, clipping at the full-scale value, and power spectra of the
   micro channels (and macro RMS) in the 15 pre-specified QC trials.
B. Original NeuroTD delay estimator on the 12 QC pairs with the two channels taken from different
   trials of the same session (trial k vs. trial k+25 mod 50). This is a coupling diagnostic: it
   shows what |delay| distribution the estimator returns when the two signals cannot share a
   trial-locked common source. It is NOT used as a null distribution for the region mean test.

Run from the repository root:
    python -u reanalysis_review2_4_2_calibration_electrodes_20261031_1134/scripts/posthoc_signal_checks.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403,E402
import common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from progress import RateTracker, Reporter, fmt_hms  # noqa: E402

TRIALS = [0, 12, 24, 37, 49]


def _delay_task(task):
    import h5py
    from neurotd.time_varying_shift import compute_windowed_mse_with_hop
    ses, ti, tj, ci, cj, label = task
    t0 = time.perf_counter()
    try:
        with h5py.File(C.NWB[ses], "r") as f:
            d = f["acquisition/ecephys.lfp/data"]
            x = np.asarray(d[ti * C.TRIAL_LEN:(ti + 1) * C.TRIAL_LEN, ci], dtype=np.float64)
            y = np.asarray(d[tj * C.TRIAL_LEN:(tj + 1) * C.TRIAL_LEN, cj], dtype=np.float64)
        _, shifts = compute_windowed_mse_with_hop(x, y, C.W, C.H, window_fun="rectangle", continuous=False)
        return dict(task=task, ok=True, shifts=np.asarray(shifts, dtype=int), seconds=time.perf_counter() - t0, error=None)
    except Exception:
        return dict(task=task, ok=False, shifts=None, seconds=time.perf_counter() - t0, error=traceback.format_exc())


def part_a(REP):
    import h5py
    from scipy.signal import welch
    rows = []
    bands = [(0, 1), (1, 30), (30, 300), (300, 3000), (3000, 16000)]
    ch = pd.read_csv(C.ELE / "channel_verification_table.csv")
    labels = ch[ch.session == "ses-01"].label.tolist()
    n = 0
    for s in (1, 2, 3):
        for t in TRIALS:
            REP.set_task(f"Amplitude, quantization, clipping and spectra for ses-0{s} trial {t}")
            with h5py.File(C.NWB[s], "r") as f:
                x = np.asarray(f["acquisition/ecephys.lfp/data"][t * C.TRIAL_LEN:(t + 1) * C.TRIAL_LEN, :])
                m = np.asarray(f["acquisition/ecephys.ieeg/data"][t * 16000:(t + 1) * 16000, :])
            fr, P = welch(x, fs=C.FS, nperseg=C.FS, axis=0)
            tot = P.sum(0)
            for c in range(32):
                v = x[:, c]
                u = np.unique(np.round(v, 9))
                step = float(np.min(np.diff(u))) if u.size > 1 else np.nan
                row = dict(session=f"ses-0{s}", trial_pos=t, channel=c, label=labels[c],
                           rms_data_units=float(v.std()), abs_max=float(np.abs(v).max()),
                           frac_at_full_scale_200=float(np.mean(np.abs(v) >= 199.999)),
                           min_step=step, step_times_32767=step * 32767 if np.isfinite(step) else np.nan,
                           macro_median_rms_data_units=float(np.median(m.std(0))))
                for lo, hi in bands:
                    row[f"power_frac_{lo}_{hi}Hz"] = float(P[(fr >= lo) & (fr < hi), c].sum() / tot[c])
                rows.append(row)
            n += 1
            REP.set_progress(done=n, total=15, unit="trials")
    df = pd.DataFrame(rows)
    df.to_csv(C.WQC / "posthoc_amplitude_quantization_spectrum.csv", index=False)
    by = df.groupby("label").agg(rms=("rms_data_units", "median"), clip_frac_max=("frac_at_full_scale_200", "max"),
                                 clip_frac_median=("frac_at_full_scale_200", "median"),
                                 lowfreq_1_30=("power_frac_1_30Hz", "median"),
                                 band_30_300=("power_frac_30_300Hz", "median"),
                                 band_300_3000=("power_frac_300_3000Hz", "median"),
                                 step32767=("step_times_32767", "median"))
    by.to_csv(C.WQC / "posthoc_amplitude_by_channel.csv")
    summ = dict(micro_rms_median=float(df.rms_data_units.median()), micro_abs_max=float(df.abs_max.max()),
                quantization_step_times_32767_median=float(df.step_times_32767.median()),
                macro_rms_median=float(df.macro_median_rms_data_units.median()),
                frac_samples_at_full_scale_overall=float(df.frac_at_full_scale_200.mean()),
                n_channel_trials_with_any_clipping=int((df.frac_at_full_scale_200 > 0).sum()),
                n_channel_trials_clip_ge_1pct=int((df.frac_at_full_scale_200 >= 0.01).sum()),
                power_frac_1_30Hz_median=float(df["power_frac_1_30Hz"].median()),
                power_frac_30_3000Hz_median=float((df["power_frac_30_300Hz"] + df["power_frac_300_3000Hz"]).median()))
    REP.line(f"[A] Median microwire RMS {summ['micro_rms_median']:.2f} (data units) | maximum absolute value {summ['micro_abs_max']:.6f} | "
             f"minimum quantization step × 32767 = {summ['quantization_step_times_32767_median']:.4f} | median macro RMS {summ['macro_rms_median']:.1f}")
    REP.line(f"[A] Fraction of samples at ±200 full scale {summ['frac_samples_at_full_scale_overall']:.2e} | channel-trials with clipping: "
             f"{summ['n_channel_trials_with_any_clipping']}/480, of which {summ['n_channel_trials_clip_ge_1pct']} have ≥1% clipped samples")
    REP.line(f"[A] Median microwire power fractions: 1–30 Hz {summ['power_frac_1_30Hz_median']:.3f}, 30–3000 Hz {summ['power_frac_30_3000Hz_median']:.3f}")
    return summ


def part_b(REP):
    sel = pd.read_csv(C.WQC / "selected_pairs.csv")
    meta = pd.read_csv(C.PREV / "tables/trial_metadata.csv")
    zw = np.load(C.PREV / "checkpoints/windows_all_trials.npz")["shifts_samples"]
    tasks = []
    for r in sel.itertuples():
        for s in (1, 2, 3):
            for t in TRIALS:
                tasks.append((s, t, (t + 25) % 50, int(r.channel_i), int(r.channel_j), f"{r.label_i}–{r.label_j}"))
    REP.line(f"[B] Original estimator (W=1001, H=500, rectangular window, continuous=False) on mismatched trials: {len(tasks)} trial-by-pair combinations, "
             f"{C.N_WORKERS} processes")
    out = []
    rt = RateTracker(min_units=C.N_WORKERS, min_seconds=10)
    last = 0
    n_fail = 0
    REP.set_progress(done=0, total=len(tasks), unit="pairs")
    with mp.get_context("spawn").Pool(C.N_WORKERS) as pool:
        it = pool.imap_unordered(_delay_task, tasks)
        k = 0
        while True:
            try:
                res = it.next(timeout=10)
            except mp.TimeoutError:
                continue
            except StopIteration:
                break
            k += 1
            s, ti, tj, ci, cj, lab = res["task"]
            if not res["ok"] or res["shifts"].size != 508:
                n_fail += 1
                REP.line(f"[B] Error: ses-0{s} {lab} trials {ti}/{tj}, recorded as failed. Cause: {(res['error'] or 'size')[-300:]}")
                continue
            sh = res["shifts"]
            tidx = int(np.flatnonzero((meta.session == f"ses-0{s}").to_numpy() & (meta.trial_pos == ti).to_numpy())[0])
            pidx = int(sel[(sel.channel_i == ci) & (sel.channel_j == cj)].pair_index.iloc[0])
            same = zw[tidx, pidx].astype(int)
            a = np.abs(sh) * C.MS_PER_SAMPLE
            b = np.abs(same) * C.MS_PER_SAMPLE
            out.append(dict(session=f"ses-0{s}", trial_i=ti, trial_j=tj, pair=lab, pair_index=pidx,
                            region=sel[sel.pair_index == pidx].region.iloc[0],
                            selection_reason=sel[sel.pair_index == pidx].selection_reason.iloc[0],
                            mismatched_zero_fraction=float(np.mean(sh == 0)),
                            mismatched_mean_abs_nonzero_ms=float(a[a > 0].mean()) if (a > 0).any() else np.nan,
                            mismatched_boundary_fraction=float(np.mean(np.abs(sh) == 500)),
                            matched_zero_fraction=float(np.mean(same == 0)),
                            matched_mean_abs_nonzero_ms=float(b[b > 0].mean()) if (b > 0).any() else np.nan,
                            matched_boundary_fraction=float(np.mean(np.abs(same) == 500)),
                            seconds=res["seconds"]))
            REP.set_progress(done=k, total=len(tasks), unit="pairs", extra=f"failed {n_fail}")
            REP.set_task(f"ses-0{s} {lab} trial {ti} × trial {tj}")
            if time.time() - last >= 10 or k == len(tasks):
                last = time.time()
                REP.line(f"[B] Completed {k}/{len(tasks)} pairs | failures {n_fail} | {rt.text(k, len(tasks), 'pairs')}")
    df = pd.DataFrame(out)
    df.to_csv(C.WQC / "posthoc_mismatched_trial_delays.csv", index=False)
    summ = df.groupby(["pair", "region", "selection_reason"]).agg(
        n=("session", "size"),
        matched_zero=("matched_zero_fraction", "mean"), mismatched_zero=("mismatched_zero_fraction", "mean"),
        matched_mean_abs_ms=("matched_mean_abs_nonzero_ms", "mean"),
        mismatched_mean_abs_ms=("mismatched_mean_abs_nonzero_ms", "mean"),
        matched_boundary=("matched_boundary_fraction", "mean"),
        mismatched_boundary=("mismatched_boundary_fraction", "mean")).reset_index()
    summ.to_csv(C.WQC / "posthoc_mismatched_trial_delays_summary.csv", index=False)
    for r in summ.itertuples():
        REP.line(f"[B] {r.pair:<10} {r.selection_reason:<10} same trial: zero fraction {r.matched_zero:.2f}, mean nonzero delay {r.matched_mean_abs_ms:.2f} ms | "
                 f"mismatched trials: zero fraction {r.mismatched_zero:.2f}, mean nonzero delay {r.mismatched_mean_abs_ms:.2f} ms")
    return dict(n_tasks=len(tasks), n_failed=n_fail)


def update_findings(a):
    F = pd.read_csv(C.ELE / "electrode_findings.csv")
    i = F.index[F.id == "E11"][0]
    F.loc[i, "category"] = "conflicting evidence"
    F.loc[i, "finding"] = (
        f"ecephys.lfp declares volts and conversion=1.0, but the minimum quantization step ×32767 is {a['quantization_step_times_32767_median']:.3f}, "
        f"with maximum absolute value {a['micro_abs_max']:.4f} and median RMS about {a['micro_rms_median']:.1f} (macro about {a['macro_rms_median']:.0f}). "
        "An RMS of several volts is physically implausible for these microwires. The numerical format is consistent with int16 scaled to ±200 full scale, possibly µV pending confirmation.")
    F.loc[i, "evidence"] = "NWB data attributes and waveform_qc/posthoc_amplitude_quantization_spectrum.csv (post hoc)"
    F.loc[i, "impact"] = ("A common linear change of units preserves the delay estimate, but clipping at ±200 changes the waveform. Report the clipping fraction. "
                          "The authors need to confirm the units and conversion.")
    extra = pd.DataFrame([dict(
        id="E13", topic="Microwire spectra and cross-bundle correlation (post hoc)", category="insufficient evidence; author confirmation needed",
        finding=(f"Median microwire power fractions were {a['power_frac_1_30Hz_median']:.3f} at 1–30 Hz and {a['power_frac_30_3000Hz_median']:.3f} at 30–3000 Hz. "
                 "After low-pass filtering at 200 Hz, zero-lag correlations between microwires in different bundles and between microwires and any macro contact (including the same-shaft first contact) were close to zero. "
                 "Neighbouring microwires within a bundle had higher correlations (0.3–0.7) and peaked at zero lag."),
        evidence="waveform_qc/micro_vs_micro_correlation_lowpass200Hz.csv, micro_vs_macro_correlation_lowpass200Hz.csv, posthoc_amplitude_quantization_spectrum.csv",
        impact=("Possible explanations include high-pass filtering, alternative or within-bundle referencing, and export conversion. "
                "These explanations require confirmation. With little shared signal, cross-bundle delay estimates could primarily reflect noise (see mismatched-trial diagnostics)."))])
    F = pd.concat([F[F.id != "E13"], extra], ignore_index=True)
    F.to_csv(C.ELE / "electrode_findings.csv", index=False)


def main():
    REP = Reporter(C.OUT, heartbeat_s=10)
    REP.line("Starting post hoc checks motivated by the raw waveforms")
    REP.start_stage("Raw waveform quality (post hoc)")
    try:
        a = part_a(REP)
        b = part_b(REP)
        update_findings(a)
        json.dump(dict(amplitude=a, mismatched=b), open(C.WQC / "posthoc_summary.json", "w"), indent=2)
        REP.end_stage("completed")
    except Exception as e:  # noqa: BLE001
        REP.line(f"Error: {e}\n{traceback.format_exc()}")
        REP.end_stage("failed", note=str(e)[:200])
        raise
    finally:
        REP.close()


if __name__ == "__main__":
    main()
