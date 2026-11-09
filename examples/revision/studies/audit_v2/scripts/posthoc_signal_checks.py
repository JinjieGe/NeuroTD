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
            REP.set_task(f"幅度/量化/截幅/频谱 ses-0{s} trial {t}")
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
            REP.set_progress(done=n, total=15, unit="试次")
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
    REP.line(f"[A] 微电极 RMS 中位数 {summ['micro_rms_median']:.2f}（数据单位）| 最大绝对值 {summ['micro_abs_max']:.6f} | "
             f"最小量化步长×32767 = {summ['quantization_step_times_32767_median']:.4f} | 宏电极 RMS 中位数 {summ['macro_rms_median']:.1f}")
    REP.line(f"[A] 处于 ±200 满量程的样本比例（全体）{summ['frac_samples_at_full_scale_overall']:.2e}；有截幅样本的通道×试次 "
             f"{summ['n_channel_trials_with_any_clipping']}/480；截幅 ≥1% 的 {summ['n_channel_trials_clip_ge_1pct']}")
    REP.line(f"[A] 微电极功率比例中位数：1–30 Hz {summ['power_frac_1_30Hz_median']:.3f}；30–3000 Hz {summ['power_frac_30_3000Hz_median']:.3f}")
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
    REP.line(f"[B] 原函数（W=1001, H=500, 矩形窗, continuous=False）在“不同试次配对”上估计延迟：{len(tasks)} 个试次×通道对，"
             f"{C.N_WORKERS} 个进程")
    out = []
    rt = RateTracker(min_units=C.N_WORKERS, min_seconds=10)
    last = 0
    n_fail = 0
    REP.set_progress(done=0, total=len(tasks), unit="对")
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
                REP.line(f"[B] 错误：ses-0{s} {lab} trial {ti}/{tj}；记录为失败。原因：{(res['error'] or 'size')[-300:]}")
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
            REP.set_progress(done=k, total=len(tasks), unit="对", extra=f"失败 {n_fail}")
            REP.set_task(f"ses-0{s} {lab} trial {ti} × trial {tj}")
            if time.time() - last >= 10 or k == len(tasks):
                last = time.time()
                REP.line(f"[B] 已完成 {k}/{len(tasks)} 对 | 失败 {n_fail} | {rt.text(k, len(tasks), '对')}")
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
        REP.line(f"[B] {r.pair:<10} {r.selection_reason:<10} 同试次：零值 {r.matched_zero:.2f} 非零均值 {r.matched_mean_abs_ms:.2f} ms | "
                 f"不同试次：零值 {r.mismatched_zero:.2f} 非零均值 {r.mismatched_mean_abs_ms:.2f} ms")
    return dict(n_tasks=len(tasks), n_failed=n_fail)


def update_findings(a):
    F = pd.read_csv(C.ELE / "electrode_findings.csv")
    i = F.index[F.id == "E11"][0]
    F.loc[i, "category"] = "存在明确冲突"
    F.loc[i, "finding"] = (
        f"ecephys.lfp 声明 unit=volts、conversion=1.0，但数值的最小量化步长 ×32767 = {a['quantization_step_times_32767_median']:.3f}，"
        f"最大绝对值 {a['micro_abs_max']:.4f}，RMS 中位数约 {a['micro_rms_median']:.1f}（宏电极约 {a['macro_rms_median']:.0f}）。"
        "按 volts 解释，微电极 RMS 为数伏，物理上不可能；数值形式与“int16 按 ±200 满量程换算”的导出一致（单位可能为 µV，未确认）。")
    F.loc[i, "evidence"] = "NWB data 属性；waveform_qc/posthoc_amplitude_quantization_spectrum.csv（事后补充检查）"
    F.loc[i, "impact"] = ("延迟估计只依赖相对波形，不受线性单位影响；但 ±200 处的截幅会改变波形，需报告截幅比例；"
                          "单位与换算需作者确认")
    extra = pd.DataFrame([dict(
        id="E13", topic="微电极信号的频谱与跨束相关（事后补充）", category="当前证据不足、需作者确认",
        finding=(f"微电极 1–30 Hz 功率占比中位数 {a['power_frac_1_30Hz_median']:.3f}，30–3000 Hz 占 {a['power_frac_30_3000Hz_median']:.3f}；"
                 "低通 200 Hz 后，跨电极束的微电极通道零时差相关接近 0，微电极与任何宏触点（包括同一电极 1 号触点）的相关也接近 0；"
                 "同一束内的相邻微电极则相关较高（0.3–0.7），且互相关峰在 0 滞后。"),
        evidence="waveform_qc/micro_vs_micro_correlation_lowpass200Hz.csv、micro_vs_macro_correlation_lowpass200Hz.csv、posthoc_amplitude_quantization_spectrum.csv",
        impact=("可能的解释包括：微电极数据经过高通或不同参考处理、微电极以束内参考记录、或导出转换问题；"
                "这些解释都未得到证实。若跨束信号没有共同成分，跨束通道对的延迟估计可能主要反映噪声而非耦合（见不同试次配对诊断）"))])
    F = pd.concat([F[F.id != "E13"], extra], ignore_index=True)
    F.to_csv(C.ELE / "electrode_findings.csv", index=False)


def main():
    REP = Reporter(C.OUT, heartbeat_s=10)
    REP.line("事后补充检查（方案外，由原始波形检查结果引出）启动")
    REP.start_stage("原始波形质量检查（事后补充）")
    try:
        a = part_a(REP)
        b = part_b(REP)
        update_findings(a)
        json.dump(dict(amplitude=a, mismatched=b), open(C.WQC / "posthoc_summary.json", "w"), indent=2)
        REP.end_stage("完成")
    except Exception as e:  # noqa: BLE001
        REP.line(f"错误：{e}\n{traceback.format_exc()}")
        REP.end_stage("失败", note=str(e)[:200])
        raise
    finally:
        REP.close()


if __name__ == "__main__":
    main()
