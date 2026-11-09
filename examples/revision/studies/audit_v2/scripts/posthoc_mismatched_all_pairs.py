"""Post-hoc coupling diagnostic on all 60 within-region pairs (not pre-specified).

For 10 trials per session (trial_pos 0, 5, ..., 45) the original NeuroTD estimator is run with channel i
from trial k and channel j from trial (k + 25) mod 50 of the same session. Region-level pooled mean |delay|
(nonzero windows) and the six region differences are compared with the matched-trial values from the
reused checkpoints on the same trials. Descriptive only; not a null distribution for the mean test.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403,E402
import common as C  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import infer as I  # noqa: E402
from posthoc_signal_checks import _delay_task  # noqa: E402
from progress import RateTracker, Reporter  # noqa: E402

TRIALS = list(range(0, 50, 5))


def main():
    REP = Reporter(C.OUT, heartbeat_s=10)
    REP.start_stage("事后耦合诊断：全部 60 对不同试次配对")
    try:
        pc = pd.read_csv(C.ELE / "pair_classification.csv")
        meta = pd.read_csv(C.PREV / "tables/trial_metadata.csv")
        zw = np.load(C.PREV / "checkpoints/windows_all_trials.npz")["shifts_samples"]
        ckdir = C.CK / "posthoc_mismatched"
        ckdir.mkdir(exist_ok=True)
        tasks = [(s, t, (t + 25) % 50, int(r.channel_i), int(r.channel_j), f"{int(r.pair_index)}")
                 for s in (1, 2, 3) for t in TRIALS for r in pc.itertuples()]
        done = {p.stem for p in ckdir.glob("*.npy")}
        key = lambda tk: f"s{tk[0]}_t{tk[1]:02d}_p{int(tk[5]):02d}"  # noqa: E731
        todo = [tk for tk in tasks if key(tk) not in done]
        REP.line(f"总任务 {len(tasks)} 个试次×通道对；断点续跑：可跳过 {len(tasks) - len(todo)}，仍需处理 {len(todo)}；{C.N_WORKERS} 进程")
        rt = RateTracker(min_units=C.N_WORKERS, min_seconds=10)
        last, k, nf = 0, 0, 0
        REP.set_progress(done=len(tasks) - len(todo), total=len(tasks), unit="对")
        with mp.get_context("spawn").Pool(C.N_WORKERS) as pool:
            it = pool.imap_unordered(_delay_task, todo)
            while True:
                try:
                    res = it.next(timeout=10)
                except mp.TimeoutError:
                    continue
                except StopIteration:
                    break
                k += 1
                if res["ok"] and res["shifts"].size == 508:
                    np.save(ckdir / f"{key(res['task'])}.npy", res["shifts"].astype(np.int16))
                else:
                    nf += 1
                    REP.line(f"错误：{res['task']}；记录为失败。{(res['error'] or 'size')[-300:]}")
                REP.set_progress(done=len(tasks) - len(todo) + k, total=len(tasks), unit="对", extra=f"失败 {nf}")
                s, ti, tj, ci, cj, pid = res["task"]
                REP.set_task(f"ses-0{s} trial {ti}×{tj} pair #{pid}")
                if time.time() - last >= 10 or k == len(todo):
                    last = time.time()
                    REP.line(f"[事后耦合诊断] 本次完成 {k}/{len(todo)} 对 | 已保存检查点 {len(tasks) - len(todo) + k - nf}/{len(tasks)} | "
                             f"失败 {nf} | {rt.text(k, len(todo), '对')}")
        # aggregate
        rows = []
        for s in (1, 2, 3):
            for t in TRIALS:
                tidx = int(np.flatnonzero((meta.session == f"ses-0{s}").to_numpy() & (meta.trial_pos == t).to_numpy())[0])
                for r in pc.itertuples():
                    p = ckdir / f"s{s}_t{t:02d}_p{int(r.pair_index):02d}.npy"
                    if not p.exists():
                        continue
                    mis = np.abs(np.load(p).astype(int)) * C.MS_PER_SAMPLE
                    mat = np.abs(zw[tidx, r.pair_index].astype(int)) * C.MS_PER_SAMPLE
                    rows.append(dict(session=f"ses-0{s}", trial_pos=t, pair_index=r.pair_index, region=r.region,
                                     same_bundle=r.same_bundle, S_mis=mis[mis > 0].sum(), N_mis=(mis > 0).sum(),
                                     S_mat=mat[mat > 0].sum(), N_mat=(mat > 0).sum(),
                                     zero_mis=(mis == 0).mean(), zero_mat=(mat == 0).mean()))
        df = pd.DataFrame(rows)
        df.to_csv(C.WQC / "posthoc_mismatched_all_pairs_by_trial_pair.csv", index=False)
        reg = df.groupby("region").agg(S_mat=("S_mat", "sum"), N_mat=("N_mat", "sum"), S_mis=("S_mis", "sum"),
                                       N_mis=("N_mis", "sum"), zero_mat=("zero_mat", "mean"), zero_mis=("zero_mis", "mean"))
        reg["pooled_matched_ms"] = reg.S_mat / reg.N_mat
        reg["pooled_mismatched_ms"] = reg.S_mis / reg.N_mis
        reg = reg.loc[I.REGIONS]
        reg.to_csv(C.WQC / "posthoc_mismatched_all_pairs_region.csv")
        comp = []
        for a, b in I.COMPARISONS:
            ra, rb = I.REGIONS[a], I.REGIONS[b]
            comp.append(dict(comparison=f"{I.SHORT[ra]} − {I.SHORT[rb]}",
                             delta_matched_ms=reg.pooled_matched_ms[ra] - reg.pooled_matched_ms[rb],
                             delta_mismatched_ms=reg.pooled_mismatched_ms[ra] - reg.pooled_mismatched_ms[rb]))
        comp = pd.DataFrame(comp)
        comp["ratio_mismatched_over_matched"] = comp.delta_mismatched_ms / comp.delta_matched_ms
        comp.to_csv(C.WQC / "posthoc_mismatched_all_pairs_differences.csv", index=False)
        for r in reg.itertuples():
            REP.line(f"  {I.SHORT[r.Index]}: 同试次池化 {r.pooled_matched_ms:.3f} ms（零值 {r.zero_mat:.3f}）| "
                     f"不同试次池化 {r.pooled_mismatched_ms:.3f} ms（零值 {r.zero_mis:.3f}）")
        for r in comp.itertuples():
            REP.line(f"  {r.comparison}: Δ 同试次 {r.delta_matched_ms:+.3f} ms | 不同试次 {r.delta_mismatched_ms:+.3f} ms")
        REP.end_stage("完成" if nf == 0 else f"完成（{nf} 个失败）")
    except Exception as e:  # noqa: BLE001
        REP.line(f"错误：{e}\n{traceback.format_exc()}")
        REP.end_stage("失败", note=str(e)[:200])
        raise
    finally:
        REP.close()


if __name__ == "__main__":
    main()
