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
    ("s1", "输入、溯源与受试者登记"),
    ("s2", "通道映射与官方资料"),
    ("s3", "全通道波形质量检查"),
    ("s4", "原分组与电极束延迟"),
    ("s5", "时间对应关系对照"),
    ("s6", "注入已知延迟"),
    ("s7", "任务阶段与宏电极"),
    ("s8", "有限统计验证"),
    ("s9", "绘图、报告及完整性检查"),
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
    REP.line(f"Python {env['python']} numpy {env['numpy']} 进程数 {N_WORKERS}")
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
    REP.line(f"DANDI 本地文件 {len(inv)} 个；主分析 3 个会话；其余 {n_extra} 个只登记、不混入推断")
    # reuse checkpoint shape
    z = np.load(PREV1 / "checkpoints/windows_all_trials.npz", allow_pickle=True)
    assert z["shifts_samples"].shape == (150, 60, 508)
    REP.line("复用第一轮窗口延迟数组 150×60×508（int16）")
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
    REP.line(f"一致性：中心 {okc} 移位 {oks} | 原函数 {t_orig:.2f}s 向量化 {t_vec:.2f}s")
    if not (okc and oks):
        raise RuntimeError("向量化实现与原函数不一致，停止")
    n_new = 100 * 150
    est = dict(verify_s=t_orig, vec_s=t_vec, new_pair_trials=n_new,
               s4_hours=n_new * t_vec / N_WORKERS / 3600,
               s5_hours=(112 * 150 * 2 + 12 * 150 * 3) * t_vec / N_WORKERS / 3600)
    (OUT / "logs/time_estimate_at_start.json").write_text(json.dumps(est, indent=2))
    REP.line(f"开始时估计：新同束延迟约 {est['s4_hours']:.2f} h；错配对照约 {est['s5_hours']:.2f} h（预计）")
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
        dict(id="E14", topic="DANDI 000574 元数据中的解剖索引", category="尚未确认",
             finding="DANDI 页面写明 This Dandiset does not specify which brain regions（无 UBERON 解剖索引）。版本 0.231010.1809，联系人 Jan Hohenheim。",
             evidence="https://dandiarchive.org/dandiset/000574/0.231010.1809", impact="不能从 DANDI 元数据确认微电极脑区"),
        dict(id="E15", topic="NWB 转换代码", category="尚未确认",
             finding="DANDI 关联的 GitHub nwb-example（janhohenheim/nwb-example）只是读取示例，不是 NIX→NWB 转换脚本。G-Node 页面本次 HTTP 403，未能下载 NIX 原始说明。",
             evidence="https://github.com/janhohenheim/nwb-example ；G-Node doi:10.12751/g-node.d76994",
             impact="无法从转换代码判定微电极是否被错误地接到宏触点行"),
        dict(id="E16", topic="AHL1–8 在微电极序列中的含义", category="存在冲突",
             finding="NWB 把 ecephys.lfp（Micro iEEG data，32 kHz，32 列）的 32 列标为 AHL1–8、AL1–8、ECL1–8、PHR1–8，且引用与宏电极 1–8 号触点相同的电极表行。Sci Data：每根宏电极有 9 根微电极从尖端伸出约 4 mm，Fig. 2 把电极尖端当作微电极位置。因此标签像沿杆宏触点，官方描述则支持尖端微丝。",
             evidence="NWB electrodes 引用；Boran et al. 2020 Sci Data Recording setup / Fig. 2",
             impact="版本 A 的 MTG/STG 分组缺乏独立支持；版本 C 用束名、不冠脑区"),
        dict(id="E17", topic="任务阶段边界", category="已确认",
             finding="注视 1 s、编码 2 s、保持 3 s、随后探测。Sci Adv 以探测为 0：fixation [−6,−5]、encoding [−5,−3]、maintenance [−3,0]。NWB 试次长 8 s，NeuroTD 用 start+6 对齐反应时，与探测在第 6 s 一致。NWB trials 表没有单独的阶段列。",
             evidence="Boran 2019 Sci Adv Methods；Sci Data Fig. 1；hiEEG_functions.py response_time 定义",
             impact="阶段分析按 0–1 / 1–3 / 3–6 / 6–8 s 切分，属文献对齐而非 NWB 字段"),
    ]
    findings = pd.concat([findings, pd.DataFrame(extra)], ignore_index=True)
    findings.to_csv(OUT / "electrodes/electrode_findings.csv", index=False)
    all_bundle_pairs().to_csv(OUT / "electrodes/bundle_pairs_112.csv", index=False)
    # Version B: not created
    (OUT / "electrodes/version_B_status.txt").write_text(
        "B 未建立。没有独立于 NWB 标签的转换代码或 NIX 通道表，足以把微电极行从宏触点行分开或改侧别。\n")
    for r in extra:
        REP.line(f"  [{r['id']}] {r['topic']}：{r['category']}")
    # answers to the five questions (pre-data-collection of this round, based on sources)
    q = Path(OUT / "electrodes/five_mapping_questions.md")
    q.write_text("\n".join([
        "# 五个映射问题（本轮查阅后）",
        "",
        "1. AHL1–AHL8 等在微电极序列中是八个微丝还是沿杆宏触点？**存在冲突。** 名称与宏触点编号一致并共用电极表行；数据描述支持尖端微丝。",
        "2. 微电极是否应按深部电极尖端定位？**官方描述支持尖端（Sci Data Fig. 2），NWB 坐标不是按尖端复制的。** 属描述与元数据冲突，不是已在 NWB 内确认。",
        "3. 电极表 location 和坐标是否只适用于宏触点？**高度可能，尚未被转换代码证实。** 微电极序列没有单独的电极表行。",
        "4. ECL、LR 左右侧不一致来自哪里？**尚未确认来源。** 只观察到 location 字符串与按 NWB 轴说明的 z 符号相反。未改侧别。",
        "5. 实际单位、滤波、参考？参考：**已确认**共同颅内参考。滤波：**存在冲突**（表 0.5–1000 Hz vs 描述 0.5–5000 Hz）。单位：**存在冲突**（声明 volts，数值像 ±200 满量程 int16）。",
        "",
        "版本选择：A 追溯原图；C 为同束中性分组（主描述）；B 不建立；D2（尖端=1 号触点 location）仅敏感性。",
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
    REP.set_progress(done=0, total=len(tasks), unit="试次")
    rt = RateTracker()
    rows = []
    with make_pool() as pool:
        for i, r in enumerate(pool.imap_unordered(_qc_task, tasks, 1), 1):
            rows.extend(r)
            REP.set_progress(done=i)
            if i % 10 == 0 or i == len(tasks):
                REP.line(f"[质量检查] 试次 {i}/{len(tasks)} | {rt.text(i, len(tasks), '试次')}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "qc/channel_trial_metrics.csv", index=False)
    summ = df.groupby("label").agg(rms_median=("rms", "median"), sat_mean=("sat_frac", "mean"),
                                   nonfinite_sum=("n_nonfinite", "sum"),
                                   longest_constant_p95=("longest_constant", lambda s: float(np.quantile(s, 0.95)))).reset_index()
    summ.to_csv(OUT / "qc/channel_summary.csv", index=False)
    REP.line(f"非有限值合计 {df.n_nonfinite.sum()}；饱和比例中位数 {df.sat_frac.median():.2e}")
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
    REP.line(f"同束 112 对中复用已有检查点 {reused} 对；需新算 {len(todo)} 个试次×对")
    # skip existing
    todo2 = [t for t in todo if not Path(t["path"]).exists()]
    REP.line(f"断点续跑：可跳过 {len(todo)-len(todo2)}，仍需 {len(todo2)}")
    rt = RateTracker(min_units=6, min_seconds=8)
    done = 0
    if todo2:
        with make_pool() as pool:
            for i, res in enumerate(pool.imap_unordered(_delay_task, todo2, 1), 1):
                done = i
                if i % 20 == 0 or i == len(todo2):
                    REP.set_progress(done=i, total=len(todo2), unit="试次×对")
                    REP.line(f"[延迟估计] 新算 {i}/{len(todo2)} | 复用 {reused} 对 | {rt.text(i, len(todo2), '对')}")
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
    REP.line(f"已写出 bundle_shifts_150x112x508.npz；复用 {reused} 对，新算检查点 {len(todo)}")


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
    REP.line(f"时间对应任务 {len(tasks)}，已有检查点 {len(tasks)-len(todo)}，仍需 {len(todo)}")
    rt = RateTracker(min_units=6, min_seconds=8)
    if todo:
        with make_pool() as pool:
            for i, _ in enumerate(pool.imap_unordered(_pair_task, todo, 1), 1):
                if i % 25 == 0 or i == len(todo):
                    REP.set_progress(done=i, total=len(todo), unit="对照估计")
                    REP.line(f"[时间对应] {i}/{len(todo)} | {rt.text(i, len(todo), '个')}")
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
    REP.line("已写出 mismatch/bundle_matched_vs_mismatch_map0.csv")


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
    REP.line(f"注入条件 {len(tasks)}，仍需 {len(todo)}")
    recs = []
    rt = RateTracker()
    with make_pool() as pool:
        # map with metadata: run sequentially collect via imap on todo, then reload
        if todo:
            for i, res in enumerate(pool.imap(_inject_task, todo), 1):
                if i % 10 == 0 or i == len(todo):
                    REP.line(f"[注入] {i}/{len(todo)} | {rt.text(i, len(todo), '条件')}")
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
    REP.line("注入实验全部条件已汇总（不挑选最好条件）")


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
        raise RuntimeError(f"宏电极序列中缺少标签 {missing}；可用 {sorted(col_of)[:12]}…")
    if any(v < 0 or v >= 48 for v in idx.values()):
        raise RuntimeError(f"宏电极列号越界: {idx}")
    REP.line("宏电极列映射 " + ", ".join(f"{k}→{v}" for k, v in sorted(idx.items())))
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
            REP.line(f"[宏电极] ses-01 试次 {t+1}/50")
    pd.DataFrame(mac_rows).to_csv(OUT / "macro/ses01_adjacent_contacts.csv", index=False)
    REP.line(f"宏电极参数 W={W_MACRO} H={H_MACRO} 搜索±{MAX_SHIFT_MACRO}（{W_MACRO/FS_MACRO*1000:.2f} ms 窗）")


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
    REP.line("真实数据点估计已写出。下列校准决定能否把 p 值当作正式显著性。")
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
    REP.line("束级试次均值 lag-1 φ=" + ", ".join(f"{b} {p:.3f}" for b, p in zip(BUNDLES, phi_hat)))
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
        REP.set_task(f"校准 {name} nsim={nsim}")
        rej, n = fwer_of("M2", phi, nsim, seed[2] + SEED * 10)
        cal_rows.append(dict(stage="calibration", method="M2_L10", scenario=name, nsim=n,
                             fwer=rej / n, fwer_events=rej, alpha=0.05))
        REP.line(f"  校准 {name} M2 L=10 FWER={rej}/{n}={rej/n:.3f}")
    val_rows = []
    specs_v = [("A_iid", 0.0, 1000, [SEED, 2, 0]), ("B2_phi0.6", 0.6, 1000, [SEED, 2, 1])]
    for name, phi, nsim, seed in specs_v:
        REP.set_task(f"独立验证 {name} nsim={nsim}")
        rej, n = fwer_of("M2", phi, nsim, seed[2] + SEED * 20)
        from scipy.stats import beta
        lo = 0 if rej == 0 else beta.ppf(0.025, rej, n - rej + 1)
        hi = 1 if rej == n else beta.ppf(0.975, rej + 1, n - rej)
        verdict = "通过" if hi <= 0.075 else ("不通过" if lo > 0.075 else "证据不足")
        val_rows.append(dict(stage="validation", method="M2_L10", scenario=name, nsim=n,
                             fwer=rej / n, fwer_events=rej, cp_low=float(lo), cp_high=float(hi),
                             verdict=verdict, alpha=0.05))
        REP.line(f"  验证 {name} M2 L=10 FWER={rej}/{n}={rej/n:.3f} CP[{lo:.3f},{hi:.3f}] {verdict}")
    pd.DataFrame(cal_rows).to_csv(OUT / "calibration/calibration_fwer.csv", index=False)
    pd.DataFrame(val_rows).to_csv(OUT / "calibration/validation_fwer.csv", index=False)
    passed = all(r["verdict"] == "通过" for r in val_rows)
    save_state(m2_l10_validation_passed=passed)
    # final decision table
    dec = pd.DataFrame(rows)
    if not passed:
        dec["decision"] = "证据不足（M2 L=10 未在独立验证核心情景通过）"
        dec["interpretation"] = "描述性差值可报告；不能作为校准后的显著性"
    else:
        dec["decision"] = ["显著（家族 0.05）" if min(6 * p, 1) < 0.05 else "不显著（不等于无差异）"
                           for p in dec["M2_L10_raw_p"]]
    # mapping restriction
    dec["anatomy"] = "不宜解释为确认的脑区差异（微电极位置与 NWB 标签冲突；版本 C 仅为电极束）"
    dec.to_csv(OUT / "realdata/final_decisions.csv", index=False)
    REP.line("正式判定已按预定规则写出：通过校准才能称显著，否则证据不足")


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
    ax.set_xlabel("同束触点编号差")
    ax.set_ylabel("非零 |delay| 均值 (ms)")
    ax.legend()
    ax.set_title("版本 C：同束绝对延迟（描述性）")
    fig.tight_layout()
    fig.savefig(FIG / "fig_bundle_delay_vs_gap.png", dpi=160)
    fig.savefig(FIG / "fig_bundle_delay_vs_gap.pdf")
    plt.close()
    mm = pd.read_csv(OUT / "mismatch/bundle_matched_vs_mismatch_map0.csv")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(mm.mismatch_mean_abs, mm.matched_mean_abs, s=12)
    lim = [0, max(mm.mismatch_mean_abs.max(), mm.matched_mean_abs.max())]
    ax.plot(lim, lim, "k-", lw=0.6)
    ax.set_xlabel("错配 |delay| 均值 (ms)")
    ax.set_ylabel("真实配对 |delay| 均值 (ms)")
    ax.set_title("真实配对 vs 会话内 set_size 错配")
    fig.tight_layout()
    fig.savefig(FIG / "fig_matched_vs_mismatch.png", dpi=160)
    fig.savefig(FIG / "fig_matched_vs_mismatch.pdf")
    plt.close()
    inj = pd.read_csv(OUT / "injection/recovery_by_condition.csv")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(range(len(inj)), inj.recover2, "o-")
    ax.set_xticks(range(len(inj)), [f"{r.tau}/m{r.mix}/a{r.amp}" for r in inj.itertuples()], rotation=75, fontsize=6)
    ax.set_ylabel("容限 2 样本内恢复比例（中位数）")
    ax.set_title("注入延迟：全部条件（不挑选）")
    fig.tight_layout()
    fig.savefig(FIG / "fig_injection_recovery.png", dpi=160)
    fig.savefig(FIG / "fig_injection_recovery.pdf")
    plt.close()
    # write reports
    _write_reports()
    expected = [
        "analysis_plan.md", "REPORT_CN.md", "SUMMARY_CN.md",
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
    REP.line(f"完整性：缺失 {missing} 空文件 {empty}")
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
    L.append("# 真实数据综合审计报告")
    L.append("")
    L.append("目录：`realdata_comprehensive_audit_20261031_2228/`。性质：前两轮之后的补充分析，不是预注册。")
    L.append("")
    L.append("## 三种结论分开")
    L.append("")
    L.append("### A. 计算层面")
    L.append("三个 NWB 按文件名加载；延迟参数与第一轮相同；60 个原区内对全部复用已核验检查点。向量化实现与原函数在基准对上中心与移位完全一致后，才用于新的同束对。程序能复现延迟数组，不等于解剖标签正确，也不等于延迟是神经传导。")
    L.append("")
    L.append("### B. 方法与信号层面")
    L.append(f"注入实验条件数见 `injection/recovery_all_conditions.csv`。按条件中位恢复比例（容限 2 样本）见该表与图。错配对照：{len(mm)} 个同束对的真实配对 vs map0 错配均值差中位数 {mm.delta_mean_abs.median():.3f} ms。")
    L.append("错配仍保留相同任务时段，因此错配均值接近真实配对时，只能说明估计量不一定依赖同一试次的精细时间对齐，不能证明不存在任何耦合。")
    L.append("")
    L.append("### C. 神经解释层面")
    L.append("微电极序列与宏电极共用电极表行（E2/E16）。MTG/STG 作为微电极脑区**不能确认**。版本 C 只用束名。正式束间比较：" + "；".join(f"{r.A}−{r.B} Δ={r.difference_ms:+.3f} ms → {r.decision}" for r in dec.itertuples()) + "。")
    L.append("")
    L.append("## 校准")
    L.append(val.to_string(index=False))
    L.append("")
    L.append(f"M2 L=10 独立验证通过：{st.get('m2_l10_validation_passed')}")
    L.append("")
    L.append("## 映射五问")
    L.append((OUT / "electrodes/five_mapping_questions.md").read_text())
    L.append("")
    L.append("## 对审稿意见")
    L.append("- 2.4.2：重叠窗口已不再当作独立观测；本轮仍无在 φ=0.6 下校准合格的显著性。")
    L.append("- 2.5：单位、滤波、微电极位置存在冲突；近零对未证实为重复通道。")
    L.append("- 2.7：不能把四脑区顺序写成确认的神经解剖差异；海马较高依赖 AHL1 组合（上一轮）。")
    L.append("")
    L.append("## 耗时")
    if len(timing):
        L.append(timing.to_string(index=False))
    (OUT / "REPORT_CN.md").write_text("\n".join(L), encoding="utf-8")
    (OUT / "SUMMARY_CN.md").write_text(
        "# 简版摘要\n\n"
        "主分析仍是 sub-01 三个会话。微电极脑区标签不可靠，主描述改为四束同束对。"
        "延迟估计可复现。注入与错配结果见报告。正式推断：" +
        ("校准未通过，六组束间差均为证据不足。" if not st.get("m2_l10_validation_passed") else "见 final_decisions.csv") +
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
        REP.line("已写出 REPORT_CN.md、SUMMARY_CN.md、英文 Methods 与回复草稿（未声称稿件已改）")


def main():
    global REP
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(k for k, _ in STAGES))
    args = ap.parse_args()
    want = [s.strip() for s in args.stages.split(",") if s.strip()]
    REP = Reporter(OUT, heartbeat_s=10)
    names = dict(STAGES)
    REP.line(f"真实数据综合审计启动 | PID {os.getpid()} | 阶段 {want} | {OUT}")
    fn = {k: globals()[f"stage_{k}"] for k, _ in STAGES}
    done, complete = [], False
    for s in want:
        REP.start_stage(names[s])
        try:
            r = fn[s]()
            if s == "s9" and r is False:
                REP.end_stage("完成（缺项）")
                done.append(s)
                continue
            REP.end_stage("完成")
            done.append(s)
            complete = s == "s9" and r is True
        except Exception as e:
            REP.line(f"[{names[s]}] 错误：{type(e).__name__}: {e}")
            REP.line(traceback.format_exc())
            REP.end_stage("失败", note=str(e)[:200])
            break
    if complete:
        REP.line("全部完成")
    else:
        left = [names[x] for x in want if x not in done]
        REP.line(f"本次结束。完成：{[names[x] for x in done]}；未完成：{left}")
    own, kids = peak_rss_mb()
    REP.line(f"峰值内存 主进程 {own:.0f} MB / 子进程 {kids:.0f} MB")
    REP.close()


if __name__ == "__main__":
    main()
