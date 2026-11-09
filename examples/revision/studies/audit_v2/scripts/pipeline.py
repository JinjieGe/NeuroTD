"""Review 2.4.2 / Fig. 4c: calibration of inference methods, electrode verification and re-test.

Run from the repository root:
    python -u reanalysis_review2_4_2_calibration_electrodes_20261031_1134/scripts/pipeline.py [--stages s1,s2,...]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import *  # noqa: F401,F403,E402
import common as C  # noqa: E402

sys.path.insert(0, str(C.ROOT / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import infer as I  # noqa: E402
import genmodels as G  # noqa: E402
from progress import RateTracker, Reporter, fmt_hms, peak_rss_mb  # noqa: E402

STAGES = [
    ("s1", "输入及旧结果核验"), ("s2", "电极元数据核验"), ("s3", "原始波形质量检查"),
    ("s4", "生成模型与程序验证"), ("s5", "校准模拟"), ("s6", "独立验证"),
    ("s7", "真实数据重新检验"), ("s8", "绘图、报告及完整性检查"),
]
REP: Reporter | None = None
ANA = I.analytic_method_names()
BOOT = I.boot_method_names()
ALL_METHODS = ANA + BOOT


# ============================================================================ worker side
_W = {}


def _init_worker(counter):
    _W["counter"] = counter
    _W["P"] = None
    _W["refs"] = {}


def _params():
    if _W.get("P") is None:
        _W["P"] = load_params()
    return _W["P"]


def _refs(n_sessions):
    if n_sessions not in _W["refs"]:
        _W["refs"][n_sessions] = load_refs(n_sessions)
    return _W["refs"][n_sessions]


def _bump(n=1):
    c = _W.get("counter")
    if c is not None:
        with c.get_lock():
            c.value += n


def make_pool(counter=None):
    ctx = mp.get_context("spawn")
    return ctx.Pool(N_WORKERS, initializer=_init_worker, initargs=(counter,))


def load_params():
    raw = json.loads((CK / "genmodel_params.json").read_text())
    return {k: (np.array(v) if isinstance(v, list) else v) for k, v in raw.items()}


def load_refs(n_sessions):
    z = np.load(CK / f"fixedb_reference_{n_sessions}sessions.npz")
    out = {}
    for det in (False, True):
        for b in I.B_VALUES:
            key = f"{'detrend' if det else 'demean'}_b{b:g}"
            out[(det, b)] = I.FixedBRef(z[key + "_body"], z[key + "_tail"], int(z[key + "_R"]))
    return out


# ---------------------------------------------------------------------------- tasks
def _ref_task(args):
    seed, n_sessions, detrend, n = args
    t0 = time.perf_counter()
    x = I.simulate_reference_chunk(np.random.SeedSequence(seed), n_sessions, 50, detrend, n)
    _bump(n)
    return dict(args=args, x=x.astype(np.float32), seconds=time.perf_counter() - t0)


def _truth_task(args):
    scen, ntype, seed, n = args
    P = _params()
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    S, N, truth = G.generate(scen, ntype, rng, n, P)
    _, d = I.influence(S, N, np.ones(S.shape[1]))
    _bump(n)
    return dict(args=args, sum=d.sum(0), sumsq=(d ** 2).sum(0), n=n, truth=truth)


def batch_counts(delta, truth, p, half95, halfB, tn):
    """Counts for one batch. p, half95, halfB: (n, M, 6)."""
    padj = np.minimum(p * I.N_COMP, 1.0)
    valid = np.isfinite(padj).all(-1) & np.isfinite(half95).all(-1)  # (n, M)
    err = np.abs(delta - truth)[:, None, :]
    rej = np.stack([(padj < a) & valid[..., None] for a in I.ALPHAS], axis=1)  # (n, A, M, 6)
    cov95 = (err <= half95) & valid[..., None]
    covB = ((err <= halfB).all(-1)) & valid
    return dict(
        n=np.array(delta.shape[0]), n_valid=valid.sum(0),
        rej=rej.sum(0),  # (A, M, 6)
        fwer=(rej[..., tn].any(-1)).sum(0),  # (A, M)
        cov95=cov95.sum(0), covB=covB.sum(0),
        sum_half95=np.where(valid[..., None], half95, 0).sum(0),
    )


def _sim_task(task):
    t0 = time.perf_counter()
    try:
        P = _params()
        rng = np.random.default_rng(np.random.SeedSequence(task["seed"]))
        S, N, truth = G.generate(task["scen"], task["ntype"], rng, task["nsim"], P)
        w = np.ones(S.shape[1])
        tn = G.true_null_mask(task["ntype"])
        if task["group"] == "ana":
            out = I.analytic_inference(S, N, w, G.SESSIONS, _refs(3))
            delta, p, h95, hB = out["delta"], out["p"], out["half95"], out["halfB"]
            nres = None
            _bump(task["nsim"])
        else:
            brng = np.random.default_rng(np.random.SeedSequence(list(task["seed"]) + [99]))
            res = []
            for k in range(task["nsim"]):
                res.append(I.boot_inference(S[k], N[k], w, G.SESSIONS, brng, task["B"]))
                _bump(1)
            delta = np.stack([r["delta"] for r in res])
            p = np.stack([r["p"] for r in res])
            h95 = np.stack([r["half95"] for r in res])
            hB = np.stack([r["halfB"] for r in res])
            nres = np.stack([r["n_valid"].min(-1) for r in res]).astype(np.int32)
        cnt = batch_counts(delta, truth, p, h95, hB, tn)
        save = {f"c_{k}": v for k, v in cnt.items()}
        save.update(truth=truth, cfg_hash=np.array(task["cfg_hash"]), seed=np.array(task["seed"]))
        if task["full"]:
            save.update(delta=delta.astype(np.float32), p=p.astype(np.float32),
                        half95=h95.astype(np.float32), halfB=hB.astype(np.float32))
            if nres is not None:
                save["n_resamples_valid_min"] = nres
        tmp = Path(str(task["path"]) + ".tmp.npz")
        np.savez_compressed(tmp, **save)
        os.replace(tmp, task["path"])
        return dict(task=task, ok=True, n_valid_min=int(cnt["n_valid"].min()), seconds=time.perf_counter() - t0,
                    error=None)
    except Exception:
        return dict(task=task, ok=False, n_valid_min=0, seconds=time.perf_counter() - t0,
                    error=traceback.format_exc())


# ============================================================================ helpers
def fmt_p(p):
    if p is None or not np.isfinite(p):
        return "NA"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def cp_interval(k, n, alpha=0.05):
    from scipy.stats import beta
    k = np.asarray(k, float)
    n = np.asarray(n, float)
    lo = np.where(k > 0, beta.ppf(alpha / 2, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta.ppf(1 - alpha / 2, k + 1, n - k), 1.0)
    return lo, hi


def write_manifest(entries: dict):
    path = OUT / "checkpoint_manifest.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(entries)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(old, indent=1, ensure_ascii=False, default=str))
    os.replace(tmp, path)


def load_trial_tables():
    trs = pd.read_csv(PREV / "tables/trial_region_summary.csv")
    pts = pd.read_csv(PREV / "tables/trial_pair_summary.csv")
    meta = pd.read_csv(PREV / "tables/trial_metadata.csv")
    return trs, pts, meta


# ============================================================================ S1
def stage_s1():
    import importlib
    env = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
           "os_cpu_count": os.cpu_count(), "workers": N_WORKERS,
           "blas_threads_per_process": os.environ.get("OMP_NUM_THREADS")}
    for m in ["numpy", "scipy", "pandas", "matplotlib", "h5py", "pynwb", "tqdm"]:
        env[m] = importlib.import_module(m).__version__
    try:
        env["hardware"] = subprocess.run(["system_profiler", "SPHardwareDataType"], capture_output=True,
                                         text=True, timeout=30).stdout
    except Exception as e:  # noqa: BLE001
        env["hardware"] = f"unavailable: {e}"
    env["git"] = "not a git repository; COMMIT UNKNOWN (provenance by SHA-256)"
    (LOGS / "environment.json").write_text(json.dumps(env, indent=2, ensure_ascii=False))
    REP.line(f"Python {sys.version.split()[0]} | numpy {env['numpy']} | scipy {env['scipy']} | pandas {env['pandas']} "
             f"| pynwb {env['pynwb']} | 进程数 {N_WORKERS}，每进程 BLAS 线程 1")

    # --- NWB hashes vs. previous provenance
    prev = pd.read_csv(PREV / "tables/provenance_nwb_sha256.csv")
    rows = []
    for s, p in NWB.items():
        size = p.stat().st_size
        REP.set_task(f"SHA-256 {p.name}")
        REP.set_progress(done=0, total=size, unit="字节")
        t0 = time.time()
        h = sha256_file(p, on_bytes=lambda d: REP.set_progress(done=d))
        old = prev[prev.session == f"ses-0{s}"].sha256.iloc[0]
        rows.append(dict(session=f"ses-0{s}", path=str(p.relative_to(ROOT)), bytes=size, sha256=h,
                         sha256_previous_round=old, match=h == old))
        REP.line(f"NWB {p.name}: {size:,} 字节 | SHA-256 {h[:16]}… | 与上一轮一致：{h == old} | {time.time() - t0:.1f} s")
    REP.set_task(None)
    nwb = pd.DataFrame(rows)
    nwb.to_csv(TAB / "provenance_nwb_sha256.csv", index=False)
    if not nwb.match.all():
        raise RuntimeError("NWB 哈希与上一轮不一致，不能复用延迟结果")

    # --- previous-round files
    files = ["checkpoints/windows_all_trials.npz", "tables/trial_region_summary.csv", "tables/trial_pair_summary.csv",
             "tables/channel_pair_mapping.csv", "tables/trial_metadata.csv", "tables/bootstrap_results_all_settings.csv",
             "tables/electrode_mapping_all_32_channels.csv", "tables/provenance_nwb_sha256.csv",
             "tables/old_checkpoint_verification.csv", "tables/checkpoint_reestimation_consistency.csv",
             "synthetic_validation/synthetic_validation_results.csv", "scripts/pipeline.py", "scripts/stats_core.py",
             "scripts/posthoc_electrode_composition.py", "REPORT_CN.md", "run.log", "timing_summary.csv"]
    man = []
    for f in files:
        p = PREV / f
        man.append(dict(file=f, exists=p.exists(), bytes=p.stat().st_size if p.exists() else None,
                        sha256=sha256_file(p) if p.exists() else None))
    pd.DataFrame(man).to_csv(TAB / "provenance_previous_round_files.csv", index=False)
    REP.line(f"上一轮文件登记哈希：{sum(m['exists'] for m in man)}/{len(man)} 存在")

    # --- window array shape and S, N reconstruction
    REP.set_task("读取 windows_all_trials.npz 并重算 S、N")
    z = np.load(PREV / "checkpoints/windows_all_trials.npz")
    Wv = z["shifts_samples"]
    sess = z["session"]
    trs, pts, meta = load_trial_tables()
    mapping = pd.read_csv(PREV / "tables/channel_pair_mapping.csv")
    reg = mapping[mapping.session == "ses-01"].region.to_numpy()
    same_map = all((mapping[mapping.session == s][["channel_i", "channel_j", "region"]].to_numpy()
                    == mapping[mapping.session == "ses-01"][["channel_i", "channel_j", "region"]].to_numpy()).all()
                   for s in ("ses-02", "ses-03"))
    absms = np.abs(Wv.astype(np.int32) * MS_PER_SAMPLE)
    nz = absms > NEAR_ZERO
    maxdiff = 0.0
    nd = 0
    for t in range(Wv.shape[0]):
        for r in I.REGIONS:
            m = reg == r
            S = absms[t, m][nz[t, m]].sum()
            N = nz[t, m].sum()
            row = trs[(trs.session == meta.session[t]) & (trs.trial_pos == meta.trial_pos[t]) & (trs.region == r)]
            maxdiff = max(maxdiff, abs(S - row.S_nonzero_ms.iloc[0]))
            nd += int(N != row.N_nonzero.iloc[0])
    checks = dict(window_shape=list(Wv.shape), shape_ok=list(Wv.shape) == [150, 60, 508],
                  sessions=sorted(set(sess.tolist())), max_abs_shift=int(np.abs(Wv).max()),
                  S_max_abs_diff_vs_table=float(maxdiff), N_mismatches_vs_table=nd,
                  pair_mapping_identical_across_sessions=bool(same_map), n_pairs_per_region=pd.Series(reg).value_counts().to_dict(),
                  total_nonzero=int(nz.sum()))
    (TAB / "input_checks.json").write_text(json.dumps(checks, indent=2, ensure_ascii=False, default=str))
    REP.line(f"窗口数组 {Wv.shape}（预期 150×60×508：{checks['shape_ok']}）| 由窗口重算 S 与汇总表最大差 {maxdiff:.2e} ms "
             f"| N 不一致 {nd} | 三会话通道对映射相同 {same_map} | 非零窗口 {checks['total_nonzero']}")
    if not checks["shape_ok"] or maxdiff > 1e-6 or nd:
        raise RuntimeError("上一轮汇总表不能由窗口检查点还原")
    REP.line("复用：9000 个试次×通道对、4,572,000 个窗口的延迟（已于上一轮从原始 NWB 抽样 96 对逐窗口复核）；本轮不重算延迟。")

    # --- previous-round errata
    bt = pd.read_csv(PREV / "tables/bootstrap_results_all_settings.csv")
    er = []
    lossB = bt[bt.bootstrap_B_used < bt.bootstrap_B]
    for r in lossB.itertuples():
        er.append(dict(issue="有效重采样次数小于 B", analysis_variant=r.analysis_variant, session_scope=r.session_scope,
                       comparison=f"{r.region_A} - {r.region_B}", block_length=r.block_length,
                       detail=f"B={r.bootstrap_B}, 有效 {r.bootstrap_B_used}, 非有限 {r.n_nonfinite_resamples}"))
    mid = bt[(bt.adjusted_p >= 0.001) & (bt.adjusted_p < 0.05)]
    for r in mid.itertuples():
        er.append(dict(issue="0.001 ≤ p_adj < 0.05（达不到 p<0.001）", analysis_variant=r.analysis_variant,
                       session_scope=r.session_scope, comparison=f"{r.region_A} - {r.region_B}",
                       block_length=r.block_length, detail=f"p_adj={r.adjusted_p:.4g}, B={r.bootstrap_B_used}"))
    syn = pd.read_csv(PREV / "synthetic_validation/synthetic_validation_results.csv")
    for r in syn[syn.check.str.contains("AR")].itertuples():
        er.append(dict(issue="校准结果被记为 passed=True", analysis_variant="synthetic", session_scope="-",
                       comparison="-", block_length="1,5",
                       detail=f"passed={r.passed}; {r.detail}。上一轮代码把这一项直接写成 True，"
                              "它是误报率超标的证据，不是通过。"))
    erdf = pd.DataFrame(er)
    erdf.to_csv(TAB / "previous_round_errata.csv", index=False)
    REP.line(f"上一轮更正事项：有效 B 不足 {len(lossB)} 行；0.001≤p_adj<0.05 共 {len(mid)} 行；"
             f"合成校准误记为通过 {int(syn.check.str.contains('AR').sum())} 项")
    save_state(prev_lossB_rows=len(lossB), prev_mid_p_rows=len(mid))

    # --- benchmarks for the time estimate
    REP.line("基准测试（真实计时，用于开始时的耗时估计）")
    trs_df = trs
    P = G.estimate_params(trs_df)
    rng = np.random.default_rng(0)
    t0 = time.time()
    S, N, _ = G.generate("C1", "G", rng, 2000, P)
    t_gen = time.time() - t0
    t0 = time.time()
    x = I.simulate_reference_chunk(1, 3, 50, False, 20000)
    t_ref = (time.time() - t0) * REF_CHUNK / 20000
    refs = {(d, b): I.build_reference(I.simulate_reference_chunk(2 + d, 3, 50, d, 20000)[:, j], n_body=1000)
            for d in (False, True) for j, b in enumerate(I.B_VALUES)}
    t0 = time.time()
    I.analytic_inference(S, N, np.ones(150), G.SESSIONS, refs)
    t_ana = (time.time() - t0) / 2000
    t0 = time.time()
    for k in range(10):
        I.boot_inference(S[k], N[k], np.ones(150), G.SESSIONS, rng, BOOT_B)
    t_boot = (time.time() - t0) / 10
    import h5py
    t0 = time.time()
    with h5py.File(NWB[1], "r") as f:
        blk = f["acquisition/ecephys.lfp/data"][0:TRIAL_LEN, :]
        mac = f["acquisition/ecephys.ieeg/data"][0:16000, :]
    t_read = time.time() - t0
    del blk, mac
    n_combo = len(G.SCENARIOS) * len(G.NULL_TYPES)
    est = {
        "per_sim_analytic_s": t_ana, "per_sim_bootstrap_8methods_B1999_s": t_boot,
        "reference_chunk_100k_s": t_ref, "generate_2000_s": t_gen, "nwb_trial_read_s": t_read,
        "workers": N_WORKERS,
        "E1_electrode_and_waveform_s": 15 * (t_read + 6.0) + 30,
        "E2a_fixedb_references_s": 6 * (REF_DRAWS / REF_CHUNK) * t_ref / N_WORKERS,
        "E2b_truth_check_and_pilot_s": (n_combo * 200_000 * t_gen / 2000 + n_combo * 100 * t_boot) / N_WORKERS,
        "E3_calibration_s": (n_combo * CAL_NSIM * (t_ana + t_boot)) / N_WORKERS,
        "E4_validation_analytic_s": n_combo * VAL_NSIM_ANA * t_ana / N_WORKERS,
        "E4_validation_bootstrap_s": n_combo * VAL_NSIM_BOOT * t_boot / N_WORKERS,
        "E5_realdata_s": 35 * t_boot * (REAL_B / BOOT_B) + 60 * t_boot * (ELEC_B / BOOT_B) / 4,
        "E6_figures_report_s": "约 2–5 分钟（粗估，未基准测试）",
    }
    boot_total = est["E4_validation_bootstrap_s"]
    est["validation_bootstrap_nsim_rule"] = (
        f"预计 {boot_total / 3600:.2f} h ≤ 6 h → 使用 {VAL_NSIM_BOOT}" if boot_total <= 6 * 3600 else
        f"预计 {boot_total / 3600:.2f} h > 6 h → 按方案改用 {VAL_NSIM_BOOT_FALLBACK}")
    save_state(val_nsim_boot=VAL_NSIM_BOOT if boot_total <= 6 * 3600 else VAL_NSIM_BOOT_FALLBACK)
    (LOGS / "time_estimate_at_start.json").write_text(json.dumps(est, indent=2, ensure_ascii=False))
    REP.line(f"  解析方法 {t_ana * 1e3:.3f} ms/次；bootstrap（8 个方法，B={BOOT_B}）{t_boot:.3f} s/次；"
             f"参考分布 {t_ref:.2f} s/10 万次；读取一个试次 32 通道+宏电极 {t_read:.2f} s")
    REP.line("开始时耗时估计（预计，6 进程）：")
    REP.line(f"  电极及原始波形检查 ≈ {fmt_hms(est['E1_electrode_and_waveform_s'])}")
    REP.line(f"  fixed-b 参考分布 ≈ {fmt_hms(est['E2a_fixedb_references_s'])}；生成模型核验+试点 ≈ {fmt_hms(est['E2b_truth_check_and_pilot_s'])}")
    REP.line(f"  校准集 ≈ {fmt_hms(est['E3_calibration_s'])}")
    REP.line(f"  独立验证：解析 ≈ {fmt_hms(est['E4_validation_analytic_s'])}；bootstrap ≈ {fmt_hms(boot_total)}"
             f"（{est['validation_bootstrap_nsim_rule']}）")
    REP.line(f"  真实数据检验 ≈ {fmt_hms(est['E5_realdata_s'])}；图表与报告 约 2–5 分钟")


# ============================================================================ S2
def lateral_of(loc):
    parts = loc.split(", ")
    return parts[1].split(" ")[0] if len(parts) > 1 else None


REG_TRANSLATION = {"Amyg": "Amygdala", "Hipp": "Hippocampus", "INS": "Insular Gyrus", "ITG": "Inferior Temporal Gyrus",
                   "MTG": "Middle Temporal Gyrus", "STG": "Superior Temporal Gyrus"}


def bundle_of(label):
    return label.rstrip("0123456789")


def stage_s2():
    import h5py
    rows, macro_rows, series = [], [], []
    for s in (1, 2, 3):
        REP.set_task(f"读取 {NWB[s].name} 电极表与 ElectricalSeries 属性")
        with h5py.File(NWB[s], "r") as f:
            e = f["general/extracellular_ephys/electrodes"]
            cols = {c: e[c][:] for c in ["id", "label", "location", "group_name", "x", "y", "z", "imp",
                                          "is_inside_soz", "filtering", "reference"]}
            axis_doc = {c: dec(e[c].attrs.get("description", "")) for c in ["x", "y", "z"]}
            groups = {k: {a: dec(v) for a, v in f[f"general/extracellular_ephys/{k}"].attrs.items()
                          if a in ("description", "location")} for k in ("ieeg", "eeg")}
            subj_desc = dec(f["general/subject/description"][()])
            for name in ("ecephys.lfp", "ecephys.ieeg", "ecephys.eeg"):
                es = f[f"acquisition/{name}"]
                ts = es["timestamps"]
                d = es["data"]
                series.append(dict(session=f"ses-0{s}", series=name, description=dec(es.attrs.get("description")),
                                   comments=dec(es.attrs.get("comments")), shape=str(d.shape), dtype=str(d.dtype),
                                   unit=dec(d.attrs.get("unit")), conversion=float(d.attrs.get("conversion")),
                                   offset=float(d.attrs.get("offset", 0.0)), resolution=float(d.attrs.get("resolution")),
                                   rate_from_timestamps_hz=float(1 / np.median(np.diff(ts[:2000]))),
                                   n_electrode_refs=int(es["electrodes"].shape[0])))
            lfp = f["acquisition/ecephys.lfp/electrodes"][:]
            ieeg = f["acquisition/ecephys.ieeg/electrodes"][:]
        lab = [dec(x) for x in cols["label"]]
        for col, ri in enumerate(lfp):
            loc = dec(cols["location"][ri])
            x, y, z = (float(cols[c][ri]) for c in ("x", "y", "z"))
            rows.append(dict(
                session=f"ses-0{s}", nwb_file=NWB[s].name, series="acquisition/ecephys.lfp (Micro iEEG data)",
                waveform_column=col, electrode_table_row=int(ri), electrode_id=int(cols["id"][ri]), label=lab[ri],
                bundle=bundle_of(lab[ri]), group_name=dec(cols["group_name"][ri]),
                group_description=groups["ieeg"].get("description"), group_location=groups["ieeg"].get("location"),
                location=loc, region=REG_TRANSLATION.get(loc.split(", ")[0], loc.split(", ")[0]),
                lateral_label=lateral_of(loc), x=x, y=y, z=z,
                axis_doc_x=axis_doc["x"], axis_doc_y=axis_doc["y"], axis_doc_z=axis_doc["z"],
                side_by_z_under_axis_doc=("Right" if z > 0 else "Left"),
                mni_x_inferred=z, mni_y_inferred=-x, mni_z_inferred=-y,
                coordinate_unit="未在 NWB 中声明；数据描述称接触点已归一化到 MNI 空间（推断为 mm，未确认）",
                coordinate_system_evidence="NWB 轴说明（+x 后、+y 下、+z 右，与 NWB 模式默认说明相同）；数据描述：ACPC 对齐后归一化到 MNI，Brainnetome 分区",
                also_referenced_by_macro_ieeg=bool(ri in set(ieeg.tolist())),
                signal_unit="volts", conversion=1.0, offset=0.0,
                filtering=dec(cols["filtering"][ri]), reference=dec(cols["reference"][ri]),
                impedance=float(cols["imp"][ri]), is_inside_soz=bool(cols["is_inside_soz"][ri])))
        if s == 1:
            for ri in ieeg:
                loc = dec(cols["location"][ri])
                x, y, z = (float(cols[c][ri]) for c in ("x", "y", "z"))
                macro_rows.append(dict(electrode_table_row=int(ri), label=lab[ri], bundle=bundle_of(lab[ri]),
                                       location=loc, lateral_label=lateral_of(loc), x=x, y=y, z=z,
                                       mni_x_inferred=z, mni_y_inferred=-x, mni_z_inferred=-y,
                                       side_by_z=("Right" if z > 0 else "Left"),
                                       is_inside_soz=bool(cols["is_inside_soz"][ri])))
            subj = subj_desc
    ch = pd.DataFrame(rows)
    ch.to_csv(ELE / "channel_verification_table.csv", index=False)
    pd.DataFrame(series).to_csv(ELE / "electrical_series_attributes.csv", index=False)
    mac = pd.DataFrame(macro_rows)
    mac.to_csv(ELE / "macro_contacts_ses-01.csv", index=False)
    REP.line(f"逐通道核验表：{len(ch)} 行（3 会话 × 32 通道）")

    # cross-session consistency
    key = ["waveform_column", "electrode_table_row", "label", "location", "x", "y", "z", "group_name", "filtering", "reference"]
    s1 = ch[ch.session == "ses-01"][key].reset_index(drop=True)
    consistent = all(ch[ch.session == s][key].reset_index(drop=True).equals(s1) for s in ("ses-02", "ses-03"))

    # contact spacing along each shaft (macro rows)
    spacing = []
    for bnd, g in mac.groupby("bundle"):
        g = g.sort_values("label", key=lambda c: c.str.extract(r"(\d+)$")[0].astype(int))
        xyz = g[["x", "y", "z"]].to_numpy()
        dd = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        spacing.append(dict(bundle=bnd, n_contacts=len(g), mean_spacing=float(dd.mean()), min=float(dd.min()),
                            max=float(dd.max()), contact1_mni_inferred=str(np.round(g.iloc[0][["mni_x_inferred", "mni_y_inferred", "mni_z_inferred"]].to_numpy(float), 1).tolist()),
                            contact1_location=g.iloc[0].location,
                            side_label_set=";".join(sorted(set(str(v) for v in g.lateral_label if v))),
                            side_by_z_set=";".join(sorted(set(g.side_by_z)))))
    sp = pd.DataFrame(spacing)
    sp.to_csv(ELE / "shaft_contact_spacing.csv", index=False)

    # pair classification (60 within-region pairs)
    pairs = pd.read_csv(PREV / "tables/channel_pair_mapping.csv")
    pairs = pairs[pairs.session == "ses-01"].reset_index(drop=True)
    pts = pd.read_csv(PREV / "tables/trial_pair_summary.csv")
    zf = pts.groupby("pair_index").apply(lambda g: g.n_zero.sum() / g.n_windows_raw.sum(), include_groups=False)
    c1 = ch[ch.session == "ses-01"].set_index("waveform_column")
    pairs["bundle_i"] = [c1.bundle[i] for i in pairs.channel_i]
    pairs["bundle_j"] = [c1.bundle[j] for j in pairs.channel_j]
    pairs["same_bundle"] = pairs.bundle_i == pairs.bundle_j
    pairs["same_side_label"] = pairs.lateral_i == pairs.lateral_j
    pairs["side_by_z_i"] = [c1.side_by_z_under_axis_doc[i] for i in pairs.channel_i]
    pairs["side_by_z_j"] = [c1.side_by_z_under_axis_doc[j] for j in pairs.channel_j]
    pairs["same_side_by_z"] = pairs.side_by_z_i == pairs.side_by_z_j
    pairs["involves_ECL"] = (pairs.bundle_i == "ECL") | (pairs.bundle_j == "ECL")
    pairs["zero_fraction_all_trials"] = [float(zf[k]) for k in pairs.pair_index]
    pairs["distance_coordinate_units"] = pairs.distance_mm
    pairs = pairs.drop(columns=["distance_mm", "session"])
    pairs.to_csv(ELE / "pair_classification.csv", index=False)
    cnt = pairs.groupby("region").agg(n_pairs=("pair_index", "size"), n_same_bundle=("same_bundle", "sum"),
                                      n_same_side_label=("same_side_label", "sum"),
                                      n_same_side_by_z=("same_side_by_z", "sum"),
                                      n_involving_ECL=("involves_ECL", "sum"),
                                      bundles=("bundle_i", lambda s: ";".join(sorted(set(s)))))
    contacts = ch[ch.session == "ses-01"].groupby("region").label.apply(lambda s: ",".join(s))
    cnt["contacts"] = [contacts.get(r, "") for r in cnt.index]
    cnt.to_csv(ELE / "region_pair_composition.csv")

    # findings
    hip = c1[c1.region == "Hippocampus"]
    ecl = c1[c1.bundle == "ECL"]
    lr = mac[mac.bundle == "LR"]
    F = []

    def add(fid, topic, category, finding, evidence, impact):
        F.append(dict(id=fid, topic=topic, category=category, finding=finding, evidence=evidence, impact=impact))

    refs_rows = sorted(c1.electrode_table_row.tolist())
    add("E1", "波形列到电极行的映射", "已确认",
        f"32 个微电极波形列通过 ecephys.lfp/electrodes 引用电极表第 {refs_rows[0]}–{refs_rows[23]} 和 {refs_rows[24]}–{refs_rows[-1]} 行；不是电极表前 32 行（前 19 行是头皮 EEG）。三个会话相同：{consistent}。",
        "NWB acquisition/ecephys.lfp/electrodes；general/extracellular_ephys/electrodes", "上一轮及原始分析的映射方式正确")
    add("E2", "微电极与宏电极共用电极表行", "存在明确冲突",
        "微电极序列（描述 'Micro iEEG data'，32 kHz）引用的 32 行，全部也被宏电极序列 ecephys.ieeg（'Macro iEEG data'，2 kHz，48 行）引用。电极表没有单独的微电极行。",
        f"NWB 引用索引；数据描述：每根宏电极有 9 根微电极，从电极尖端伸出约 4 mm（{DESCRIPTOR_URL}）",
        "微电极通道的 location、x/y/z 是同编号宏触点的元数据，不是微电极自身的位置")
    add("E3", "微电极通道的脑区标签", "存在明确冲突",
        "同一电极上的 8 个微电极通道被赋予沿电极杆分布的不同脑区（例如 AHL1–2 海马、AHL5 颞上回、AHL6–8 颞中回）；按数据描述，同一束微电极都位于该电极尖端附近（相距毫米级）。图 4c 的四个“脑区”正是由这些标签定义的。",
        "电极表 location；宏触点间距（见 shaft_contact_spacing.csv）；数据描述 Recording setup 与 Fig. 2（电极尖端=微电极位置）",
        "MTG、STG 以及部分 Amygdala、Hippocampus 微电极通道的脑区归属缺乏支持；区域比较实际上是不同触点组合的比较")
    add("E4", "坐标单位与坐标系", "当前证据不足、需作者确认",
        "x/y/z 没有单位属性；轴说明为 NWB 模式默认文字（+x 后、+y 下、+z 右）。按这一轴说明换算后（MNI_x=z, MNI_y=−x, MNI_z=−y），各电极 1 号触点的位置与其标注结构大致相符；相邻宏触点平均间距 "
        f"{sp.mean_spacing.mean():.2f} 个单位（数据描述：触点中心间距 5 mm，归一化后可有缩放）。",
        "NWB 属性；数据描述 Depth electrode localization", "距离可视为“近似毫米、MNI 归一化后”，不能作为个体解剖毫米；不据此判定标签错误")
    add("E5", "AHL1、AHL2、PHR1（海马通道）", "存在明确冲突",
        f"AHL1、AHL2 属于 AHL 电极（location：左侧海马，z<0 左侧）；PHR1 属于 PHR 电极（location：右侧海马，z>0 右侧）。海马 3 个通道对：AHL1–AHL2 同束同侧；AHL1–PHR1、AHL2–PHR1 跨半球跨电极。按数据描述，AHL3–AHL8 微电极与 AHL1、AHL2 同在 AHL 尖端，但被标为 unspecific/STG/MTG。",
        "电极表；数据描述", "“海马”组只包含 3 个通道且 2/3 的通道对跨半球；结论依赖这一组合")
    add("E6", "ECL 电极侧别", "存在明确冲突",
        f"ECL1–ECL8 的 location 均写 Right（Amygdala/Insular/STG/MTG），但 z 全部 <0（按 NWB 轴说明为左侧；z 范围 {ecl.z.min():.1f} 到 {ecl.z.max():.1f}）；电极名 ECL 的末字母与左侧命名一致；受试者描述列出的深部电极为 {subj.split('Depth electrodes: ')[-1].splitlines()[0]}。",
        "电极表 location 与 z；subject/description", "不能仅凭命名或坐标改侧别；ECL 参与的跨侧/同侧分类不可靠")
    add("E7", "LR 电极侧别（只在宏电极数据中）", "存在明确冲突",
        f"LR1–LR8 的 location 写 Left，但 z 全部 >0（{lr.z.min():.1f} 到 {lr.z.max():.1f}），与 ECL 的情况恰好相反；LR 不在本分析的微电极数据中。",
        "电极表", "提示 location 字符串的侧别可能整体互换，需作者确认")
    add("E8", "SOZ 记录", "当前证据不足、需作者确认",
        "subject/description 写 'Electrodes in seizure onset zone (SOZ): AHR,LR'，但深部电极列表中没有 AHR；电极表 is_inside_soz 只对 LR 为真。",
        "subject/description；电极表 is_inside_soz", "本分析的 4 根微电极束（AHL、AL、ECL、PHR）在表中都不在 SOZ；AHR 的含义需确认")
    add("E9", "滤波说明", "存在明确冲突",
        "电极表 filtering 对这些行写 'Passband, 0.5 to 1000 Hz'；数据描述称 ATLAS 记录通带 0.5–5000 Hz、微电极 32 kHz。微电极序列本身没有单独的滤波说明。",
        "电极表 filtering；数据描述 Recording setup", "微电极信号的实际滤波未知；不影响延迟估计的再现，但影响信号解释")
    add("E10", "参考方式", "已确认",
        "电极表 reference 为 'Common intracranial reference'，与数据描述一致（所有通道对共同颅内参考）。",
        "电极表；数据描述", "共同参考会在通道间引入共同成分，可能使近邻通道的零时差相关升高（见波形检查）")
    add("E11", "信号单位、conversion、offset", "已确认",
        "ecephys.lfp：unit=volts，conversion=1.0，offset=0.0，resolution=−1（未知），float64；时间戳给出 32 kHz。",
        "NWB data 属性", "幅度量级在波形检查中复核")
    add("E12", "三个会话的相同标签是否对应相同触点", "已确认（元数据层面）",
        f"三个会话的电极表、引用索引、标签、location、坐标完全相同：{consistent}。物理同一性只能由元数据支持。",
        "三个 NWB 文件逐字段比较", "会话间可按标签对应")
    pd.DataFrame(F).to_csv(ELE / "electrode_findings.csv", index=False)
    for fdx in F:
        REP.line(f"  [{fdx['id']}] {fdx['topic']}：{fdx['category']}")
    REP.set_task(None)
    save_state(electrode_tables_consistent=bool(consistent))


# ============================================================================ S3
def run_lengths_fraction(x, min_run=32):
    same = np.r_[False, x[1:] == x[:-1]]
    idx = np.flatnonzero(~same)
    lengths = np.diff(np.r_[idx, x.size])
    return float(lengths[lengths >= min_run].sum() / x.size), int(lengths.max())


def xcorr_peak(x, y, maxlag=500):
    n = x.size
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    X = np.fft.rfft(x - x.mean(), nfft)
    Y = np.fft.rfft(y - y.mean(), nfft)
    cc = np.fft.irfft(np.conj(X) * Y, nfft)
    lags = np.r_[np.arange(0, maxlag + 1), np.arange(-maxlag, 0)]
    vals = np.r_[cc[: maxlag + 1], cc[-maxlag:]]
    k = int(np.argmax(vals))
    denom = np.sqrt(((x - x.mean()) ** 2).sum() * ((y - y.mean()) ** 2).sum())
    return int(lags[k]), float(vals[k] / denom)


def stage_s3():
    import h5py
    from scipy.signal import butter, sosfiltfilt
    pc = pd.read_csv(ELE / "pair_classification.csv")
    sel = []
    for r in pc[pc.zero_fraction_all_trials >= 0.70].itertuples():
        sel.append((r.pair_index, "高零延迟（≥70%）"))
    for r in pc[pc.region == "Hippocampus"].itertuples():
        if r.pair_index not in [s[0] for s in sel]:
            sel.append((r.pair_index, "海马通道对"))
    for reg, g in pc.groupby("region"):
        g2 = g[g.zero_fraction_all_trials < 0.20]
        if len(g2):
            med = g2.zero_fraction_all_trials.median()
            k = int(g2.iloc[(g2.zero_fraction_all_trials - med).abs().argsort().iloc[0]].pair_index)
            if k not in [s[0] for s in sel]:
                sel.append((k, "正常对照"))
    selp = pc.set_index("pair_index").loc[[s[0] for s in sel]].reset_index()
    selp["selection_reason"] = [s[1] for s in sel]
    selp.to_csv(WQC / "selected_pairs.csv", index=False)
    REP.line(f"按预定规则选取 {len(selp)} 个通道对：" + "; ".join(
        f"{r.label_i}–{r.label_j}({r.selection_reason})" for r in selp.itertuples()))
    trials = [0, 12, 24, 37, 49]
    tasks = [(s, t) for s in (1, 2, 3) for t in trials]
    REP.set_progress(done=0, total=len(tasks), unit="试次")
    sos_micro = butter(4, 200, fs=FS, output="sos")
    sos_macro = butter(4, 200, fs=2000, output="sos")
    ch = pd.read_csv(ELE / "channel_verification_table.csv")
    c1 = ch[ch.session == "ses-01"].reset_index(drop=True)
    mac = pd.read_csv(ELE / "macro_contacts_ses-01.csv")
    pair_rows, chan_rows, align_rows = [], [], []
    corr_mm = np.zeros((32, 48))
    corr_uu = np.zeros((32, 32))
    traces = {}
    rt = RateTracker(min_units=2, min_seconds=5)
    zw = np.load(PREV / "checkpoints/windows_all_trials.npz")["shifts_samples"]
    meta = pd.read_csv(PREV / "tables/trial_metadata.csv")
    for n_done, (s, t) in enumerate(tasks):
        REP.set_task(f"ses-0{s} trial_pos={t}：读取 32 通道微电极与 48 通道宏电极")
        with h5py.File(NWB[s], "r") as f:
            blk = np.asarray(f["acquisition/ecephys.lfp/data"][t * TRIAL_LEN:(t + 1) * TRIAL_LEN, :], dtype=np.float64)
            ts_u = f["acquisition/ecephys.lfp/timestamps"][t * TRIAL_LEN:(t + 1) * TRIAL_LEN]
            ts_m_all = f["acquisition/ecephys.ieeg/timestamps"]
            m0 = t * 16000
            ts_m = ts_m_all[m0:m0 + 16000]
            mb = np.asarray(f["acquisition/ecephys.ieeg/data"][m0:m0 + 16000, :], dtype=np.float64)
        align_rows.append(dict(session=f"ses-0{s}", trial_pos=t, micro_t0=float(ts_u[0]), macro_t0=float(ts_m[0]),
                               dt0_ms=1000 * float(ts_m[0] - ts_u[0]), micro_t_end=float(ts_u[-1]),
                               macro_t_end=float(ts_m[-1]), dt_end_ms=1000 * float(ts_m[-1] - ts_u[-1])))
        REP.set_task(f"ses-0{s} trial_pos={t}：逐通道与逐通道对指标")
        for c in range(32):
            x = blk[:, c]
            fr, mr = run_lengths_fraction(x)
            chan_rows.append(dict(session=f"ses-0{s}", trial_pos=t, channel=c, label=c1.label[c],
                                  rms_uV=float(np.sqrt(np.mean((x - x.mean()) ** 2)) * 1e6),
                                  n_nonfinite=int((~np.isfinite(x)).sum()), flat_fraction_runs_ge_1ms=fr,
                                  max_identical_run=mr,
                                  saturation_fraction=float(((x == x.max()) | (x == x.min())).mean())))
        tidx = int(np.flatnonzero((meta.session == f"ses-0{s}").to_numpy() & (meta.trial_pos == t).to_numpy())[0])
        for r in selp.itertuples():
            x = blk[:, r.channel_i]
            y = blk[:, r.channel_j]
            rx = np.sqrt(np.mean((x - x.mean()) ** 2))
            ry = np.sqrt(np.mean((y - y.mean()) ** 2))
            dlt = (x - x.mean()) - (y - y.mean())
            lag, pk = xcorr_peak(x, y)
            wz = zw[tidx, r.pair_index]
            pair_rows.append(dict(session=f"ses-0{s}", trial_pos=t, pair_index=r.pair_index, region=r.region,
                                  pair=f"{r.label_i}–{r.label_j}", selection_reason=r.selection_reason,
                                  identical_fraction=float(np.mean(x == y)), r_zero_lag=float(np.corrcoef(x, y)[0, 1]),
                                  diff_rms_over_mean_rms=float(np.sqrt(np.mean(dlt ** 2)) / ((rx + ry) / 2)),
                                  rms_i_uV=rx * 1e6, rms_j_uV=ry * 1e6, xcorr_peak_lag_samples=lag,
                                  xcorr_peak_r=pk, window_zero_fraction=float(np.mean(wz == 0)),
                                  window_median_abs_shift_ms=float(np.median(np.abs(wz)) * MS_PER_SAMPLE)))
            if s == 1 and t == 0:
                a = 4 * FS
                traces[f"{r.label_i}-{r.label_j}"] = np.stack([x[a:a + 640], y[a:a + 640]]).astype(np.float32)
        # micro (low-passed, 2 kHz) vs macro correlation
        ul = sosfiltfilt(sos_micro, blk - blk.mean(0), axis=0)[::16]
        ml = sosfiltfilt(sos_macro, mb - mb.mean(0), axis=0)
        n = min(ul.shape[0], ml.shape[0])
        uz = (ul[:n] - ul[:n].mean(0)) / ul[:n].std(0)
        mz = (ml[:n] - ml[:n].mean(0)) / ml[:n].std(0)
        corr_mm += np.arctanh(np.clip(uz.T @ mz / n, -0.9999, 0.9999))
        corr_uu += np.arctanh(np.clip(uz.T @ uz / n, -0.9999, 0.9999))
        del blk, mb, ul, ml
        REP.set_progress(done=n_done + 1)
        REP.line(f"[原始波形质量检查] 会话 {s}/3 | 试次 {t} | 已完成 {n_done + 1}/{len(tasks)} 个试次 | "
                 f"{rt.text(n_done + 1, len(tasks), '试次')}")
    corr_mm = np.tanh(corr_mm / len(tasks))
    corr_uu = np.tanh(corr_uu / len(tasks))
    np.fill_diagonal(corr_uu, 1.0)
    pr = pd.DataFrame(pair_rows)
    pr.to_csv(WQC / "pair_metrics_by_trial.csv", index=False)
    cr = pd.DataFrame(chan_rows)
    cr.to_csv(WQC / "channel_metrics_by_trial.csv", index=False)
    pd.DataFrame(align_rows).to_csv(WQC / "micro_macro_time_alignment.csv", index=False)
    np.savez_compressed(WQC / "example_traces_ses-01_trial00_4s_20ms.npz", **traces)
    mlab = mac.label.tolist()
    mm = pd.DataFrame(corr_mm, index=c1.label, columns=mlab)
    mm.to_csv(WQC / "micro_vs_macro_correlation_lowpass200Hz.csv")
    pd.DataFrame(corr_uu, index=c1.label, columns=c1.label).to_csv(WQC / "micro_vs_micro_correlation_lowpass200Hz.csv")
    best = []
    for i, lab in enumerate(c1.label):
        bnd = bundle_of(lab)
        same = [j for j, m in enumerate(mlab) if bundle_of(m) == bnd]
        vals = corr_mm[i, same]
        best.append(dict(micro_channel=lab, bundle=bnd, micro_number=int(lab[len(bnd):]),
                         best_macro_same_bundle=mlab[same[int(np.argmax(vals))]], best_r=float(vals.max()),
                         r_with_macro_same_number=float(corr_mm[i, mlab.index(lab)]),
                         r_with_macro_contact1=float(corr_mm[i, mlab.index(bnd + "1")]),
                         best_macro_any=mlab[int(np.argmax(corr_mm[i]))]))
    best = pd.DataFrame(best)
    best.to_csv(WQC / "micro_best_macro_match.csv", index=False)
    same_b = np.array([[bundle_of(a) == bundle_of(b) for b in c1.label] for a in c1.label])
    off = ~np.eye(32, dtype=bool)
    uu_summary = dict(mean_r_within_bundle=float(corr_uu[same_b & off].mean()),
                      mean_r_across_bundle=float(corr_uu[~same_b].mean()),
                      n_micro_best_matches_contact1=int((best.best_macro_same_bundle.str[-1] == "1").sum()),
                      n_micro_best_matches_same_number=int((best.best_macro_same_bundle == best.micro_channel).sum()))
    REP.line(f"微电极与宏电极（低通 200 Hz）：同束微电极平均 r={uu_summary['mean_r_within_bundle']:.3f}，跨束 "
             f"{uu_summary['mean_r_across_bundle']:.3f}；32 个微电极通道中最相关宏触点为同束 1 号触点的 "
             f"{uu_summary['n_micro_best_matches_contact1']} 个，为同编号触点的 {uu_summary['n_micro_best_matches_same_number']} 个")

    # persistence of zero delays over all trials
    pers = []
    for r in selp.itertuples():
        zfr = (zw[:, r.pair_index] == 0).mean(1)
        for s in ("ses-01", "ses-02", "ses-03"):
            m = (meta.session == s).to_numpy()
            pers.append(dict(pair=f"{r.label_i}–{r.label_j}", pair_index=r.pair_index, session=s,
                             mean_zero_fraction=float(zfr[m].mean()), min_zero_fraction=float(zfr[m].min()),
                             max_zero_fraction=float(zfr[m].max()),
                             trials_ge_0_7=int((zfr[m] >= 0.7).sum()), n_trials=int(m.sum())))
    pd.DataFrame(pers).to_csv(WQC / "zero_delay_persistence.csv", index=False)

    # summary + pre-specified exclusion rule
    summ = pr.groupby(["pair_index", "pair", "region", "selection_reason"]).agg(
        identical_fraction_median=("identical_fraction", "median"), r_zero_lag_median=("r_zero_lag", "median"),
        diff_rms_ratio_median=("diff_rms_over_mean_rms", "median"),
        xcorr_lag_median=("xcorr_peak_lag_samples", "median"),
        xcorr_lag_zero_trials=("xcorr_peak_lag_samples", lambda v: int((v == 0).sum())),
        window_zero_fraction_mean=("window_zero_fraction", "mean"),
        trials_identical_ge_0_99=("identical_fraction", lambda v: int((v >= 0.99).sum())),
        n_trials=("identical_fraction", "size")).reset_index()
    bad_ch = cr.assign(bad=(cr.n_nonfinite / TRIAL_LEN >= 0.5) | (cr.flat_fraction_runs_ge_1ms >= 0.5) |
                       (cr.saturation_fraction >= 0.5)).groupby("channel").bad.mean()
    excl = []
    for r in summ.itertuples():
        p = selp[selp.pair_index == r.pair_index].iloc[0]
        rule1 = r.trials_identical_ge_0_99 >= 0.5 * r.n_trials
        rule2 = bad_ch.get(p.channel_i, 0) >= 0.5 or bad_ch.get(p.channel_j, 0) >= 0.5
        excl.append(dict(pair_index=r.pair_index, pair=r.pair, rule1_identical=bool(rule1),
                         rule2_flat_saturated_missing=bool(rule2), exclude_in_evidence_corrected=bool(rule1 or rule2)))
    excl = pd.DataFrame(excl)
    summ = summ.merge(excl, on=["pair_index", "pair"])
    summ.to_csv(WQC / "pair_metrics_summary.csv", index=False)
    json.dump(uu_summary, open(WQC / "micro_macro_summary.json", "w"), indent=2)
    n_ex = int(excl.exclude_in_evidence_corrected.sum())
    REP.line(f"排除规则（逐样本相同 ≥0.99 或恒值/饱和/缺失 ≥50%，均需 ≥50% 被检查试次）：满足 {n_ex} 对")
    for r in summ.itertuples():
        REP.line(f"  {r.pair:<12} {r.selection_reason:<10} 相同比例 {r.identical_fraction_median:.3f} | r0 {r.r_zero_lag_median:.3f} | "
                 f"差值RMS/均RMS {r.diff_rms_ratio_median:.3f} | 互相关峰滞后中位数 {r.xcorr_lag_median:+.0f} | "
                 f"零延迟窗口 {r.window_zero_fraction_mean:.2f} | 排除 {r.exclude_in_evidence_corrected}")
    save_state(n_pairs_excluded_evidence=n_ex, excluded_pairs=excl[excl.exclude_in_evidence_corrected].pair.tolist())
    REP.set_task(None)


# ============================================================================ S4
def old_method_theory():
    """Expected CBB variance relative to the true variance of a session total under AR(1), n=50, demeaned."""
    rows = []
    n = 50
    for phi in (0.0, 0.3, 0.6, 0.85):
        idx = np.arange(n)
        Sig = phi ** np.abs(idx[:, None] - idx[None, :])
        true_var = Sig.sum()
        Cm = np.eye(n) - np.ones((n, n)) / n
        Er = Cm @ Sig @ Cm
        for L in I.LS:
            V = 0.0
            for j in range(-(L - 1), L):
                wj = 1 - abs(j) / L
                V += wj * sum(Er[t, (t + j) % n] for t in range(n))
            ratio = V / true_var
            from scipy.stats import norm
            size_single = 2 * (1 - norm.cdf(1.96 * np.sqrt(ratio)))
            size_bonf = 2 * (1 - norm.cdf(norm.ppf(1 - 0.05 / 12) * np.sqrt(ratio)))
            rows.append(dict(phi=phi, L=L, expected_cbb_var_over_true_var=ratio, se_ratio=np.sqrt(ratio),
                             approx_rejection_at_nominal_0_05_single_test=size_single,
                             approx_rejection_at_bonferroni_0_05_over_6_single=size_bonf))
    return pd.DataFrame(rows)


def stage_s4():
    trs, pts, meta = load_trial_tables()
    P = G.estimate_params(trs)
    (CK / "genmodel_params.json").write_text(json.dumps({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                                          for k, v in P.items()}, indent=1))
    REP.line("生成模型参数（来自真实试次级数据）：lag-1 φ=" + ", ".join(f"{I.SHORT[r]} {p:+.3f}" for r, p in zip(I.REGIONS, P["phi"]))
             + " | N 斜率 " + ", ".join(f"{v:+.4f}" for v in P["slope_N"]))

    # --- fixed-b references
    counter = mp.get_context("spawn").Value("l", 0)
    designs = [(ns, det) for ns in (3, 2, 1) for det in (False, True)]
    tasks = []
    for ns, det in designs:
        for c in range(REF_DRAWS // REF_CHUNK):
            tasks.append(([SEED, 4, ns, int(det), c], ns, det, REF_CHUNK))
    todo = [d for d in designs if not (CK / f"fixedb_reference_{d[0]}sessions.npz").exists()]
    if any(ns for ns, _ in todo):
        need = {ns for ns, _ in todo}
        tasks = [t for t in tasks if t[1] in need]
        total = len(tasks) * REF_CHUNK
        REP.line(f"fixed-b 参考分布：{len(need)} 个设计（会话数 {sorted(need)}）× 2（去均值/去趋势）× {REF_DRAWS:,} 次 = {total:,} 次模拟")
        acc = {}
        rt = RateTracker(min_units=N_WORKERS, min_seconds=10)
        last = 0
        with make_pool(counter) as pool:
            for k, res in enumerate(pool.imap_unordered(_ref_task, tasks)):
                _, ns, det, n = res["args"]
                acc.setdefault((ns, det), []).append(res["x"])
                REP.set_progress(done=counter.value, total=total, unit="次")
                REP.set_task(f"参考分布 会话数={ns} {'去趋势' if det else '去均值'}")
                if time.time() - last >= 10 or k + 1 == len(tasks):
                    last = time.time()
                    REP.line(f"[生成模型与程序验证 / fixed-b 参考分布] 已完成 {counter.value:,}/{total:,} 次 | "
                             f"块 {k + 1}/{len(tasks)} | {rt.text(k + 1, len(tasks), '块')}")
        for ns in sorted(need):
            save = {}
            for det in (False, True):
                x = np.concatenate(acc[(ns, det)])
                for j, b in enumerate(I.B_VALUES):
                    ref = I.build_reference(x[:, j])
                    key = f"{'detrend' if det else 'demean'}_b{b:g}"
                    save[key + "_body"] = ref.body
                    save[key + "_tail"] = ref.tail
                    save[key + "_R"] = np.array(ref.R)
            np.savez_compressed(CK / f"fixedb_reference_{ns}sessions.npz", **save)
            REP.line(f"检查点：已保存 checkpoints/fixedb_reference_{ns}sessions.npz，成功")
    else:
        REP.line("fixed-b 参考分布：检查点已存在，跳过（断点续跑）")
    refq = []
    for ns in (3, 2, 1):
        refs = load_refs(ns)
        for (det, b), ref in refs.items():
            refq.append(dict(n_sessions=ns, residualization="detrend" if det else "demean", b=b, M=int(round(b * 50)),
                             R=ref.R, q95_abs_t=ref.quantile(0.95), q99_abs_t=ref.quantile(0.99),
                             q_bonf_0_05=ref.quantile(1 - 0.05 / 6), q_bonf_0_001=ref.quantile(1 - 0.001 / 6)))
    refq = pd.DataFrame(refq)
    refq.to_csv(TAB / "fixedb_reference_quantiles.csv", index=False)

    # --- program checks
    rows = []

    def rec(name, ok, detail):
        rows.append(dict(check=name, passed=bool(ok), detail=detail))
        REP.line(f"  [程序核验] {name}: {'通过' if ok else '未通过'} | {detail}")

    rng = np.random.default_rng(np.random.SeedSequence([SEED, 5, 0]))
    r = rng.standard_normal((150, 6))
    ses = G.SESSIONS
    g = I.autocov_sums(r, ses)
    naive = np.zeros((I.MAX_LAG, 6))
    for s0, s1 in ses:
        for j in range(I.MAX_LAG):
            for t in range(s0, s1 - j):
                naive[j] += r[t] * r[t + j]
    rec("自协方差和与逐项循环一致", np.allclose(g, naive), f"max|差|={np.abs(g - naive).max():.1e}")
    rr = I._residualize(r[None], ses, np.ones(150), False)[0]
    Vb = I.circular_bartlett_total(rr, ses, 5)
    bs = sum(((I._circular_block_sums(rr[s0:s1], 5)) ** 2).sum(0) / 5 for s0, s1 in ses)
    rec("圆形 Bartlett 方差 = CBB 重采样总和方差的解析值", np.allclose(Vb, bs), f"L=5，max|差|={np.abs(Vb - bs).max():.1e}")
    # Monte Carlo check of the CBB variance
    idx = I.stratified_indices(rng, ses, 5, 200_000)
    mc = rr[idx].sum(1).var(0)
    rec("圆形 Bartlett 方差与 20 万次 CBB 重采样方差一致（±2%）", np.all(np.abs(mc / Vb - 1) < 0.02),
        f"比值范围 {np.round(mc / Vb, 4).min()}–{np.round(mc / Vb, 4).max()}")
    # identical regions
    refs3 = load_refs(3)
    S1 = rng.gamma(5, 2, (150, 1)) * 1000
    N1 = rng.integers(500, 1500, (150, 1)).astype(float)
    S = np.repeat(S1, 4, 1)
    N = np.repeat(N1, 4, 1)
    a = I.analytic_inference(S[None], N[None], np.ones(150), ses, refs3)
    bo = I.boot_inference(S, N, np.ones(150), ses, rng, 999)
    rec("四脑区数据相同 → Δ̂=0、M1/M3 的 t=0 且 p=1；bootstrap p=1", np.all(a["delta"] == 0) and np.allclose(a["p"], 1)
        and np.allclose(bo["p"][:4], 1), f"Δ̂={a['delta'][0, 0]}, p(M1)={a['p'][0, 0, 0]:.3f}, p(M0)={bo['p'][0, 0]:.3f}")
    # swap order
    S = rng.gamma(5, 2, (150, 4)) * 1000
    N = rng.integers(500, 1500, (150, 4)).astype(float)
    a1 = I.analytic_inference(S[None], N[None], np.ones(150), ses, refs3)
    a2 = I.analytic_inference(S[None, :, ::-1], N[None, :, ::-1], np.ones(150), ses, refs3)
    # comparison (0,1) in original equals -(comparison (2,3)) in reversed order (regions 3,2 -> original 1,0)
    rec("交换比较顺序只改变差值符号，p 和标准误不变", np.isclose(a1["delta"][0, 0], -a2["delta"][0, 5]) and
        np.allclose(a1["p"][0, :, 0], a2["p"][0, :, 5]) and np.allclose(a1["se"][0, :, 0], a2["se"][0, :, 5]),
        f"Δ {a1['delta'][0, 0]:.4f} vs {a2['delta'][0, 5]:.4f}")
    # zero counts / missing region
    S0 = S.copy(); N0 = N.copy()
    S0[3, 1] = 0; N0[3, 1] = 0
    mu0, Nw0 = I.pooled(S0, N0, np.ones(150))
    z31 = (S0[3, 1] - mu0[1] * N0[3, 1]) / Nw0[1]
    rec("N=0 的试次对该脑区的影响值为 0（不贡献）", z31 == 0 and
        np.isclose(mu0[1], S0[:, 1].sum() / N0[:, 1].sum()), f"z={z31}；池化均值与直接计算一致")
    S0[:, 2] = 0; N0[:, 2] = 0
    a0 = I.analytic_inference(S0[None], N0[None], np.ones(150), ses, refs3)
    bad = [k for k, (x, y) in enumerate(I.COMPARISONS) if 2 in (x, y)]
    rec("整个脑区 N=0 → 涉及它的比较 Δ̂ 与 p 为 NaN（计为无效，不当作 0）",
        np.all(np.isnan(a0["delta"][0, bad])) and np.all(np.isnan(a0["p"][0, :, bad])), f"无效比较 {bad}")
    # detrend residualization
    tr = np.tile(np.arange(50.0), 3)[:, None] * np.ones((1, 6))
    res_d = I._residualize(tr[None], ses, np.ones(150), True)
    rec("去趋势残差对纯线性趋势为 0", np.abs(res_d).max() < 1e-9, f"max|残差|={np.abs(res_d).max():.1e}")
    # weights = deletion
    w = np.ones(150); w[[4, 60]] = 0
    keep = w > 0
    rec("伪迹权重 0 等价于删除该试次的 S、N", np.allclose(I.pooled(S, N, w)[0], S[keep].sum(0) / N[keep].sum(0)), "四脑区一致")
    # seeds
    x1 = G.generate("C1", "G", np.random.default_rng(np.random.SeedSequence([SEED, 1, 3, 0, 0])), 5, P)[0]
    x2 = G.generate("C1", "G", np.random.default_rng(np.random.SeedSequence([SEED, 1, 3, 0, 0])), 5, P)[0]
    rec("相同种子生成相同数据", np.array_equal(x1, x2), "逐元素相同")
    # within-session blocks
    idx = I.stratified_indices(np.random.default_rng(1), ses, 5, 2000)
    ok = all(((idx[:, i * 50:(i + 1) * 50] >= a_) & (idx[:, i * 50:(i + 1) * 50] < b_)).all() for i, (a_, b_) in enumerate(ses))
    rec("重采样块不跨会话", ok, "2000 次，L=5")
    # KV asymptotic comparison
    r1 = load_refs(1)
    kv = lambda b: 1.96 + 2.9694 * b + 0.4160 * b ** 2 - 0.5324 * b ** 3  # noqa: E731
    det_ = [f"b={b}: 有限样本 {r1[(False, b)].quantile(0.95):.3f} vs 渐近 {kv(b):.3f}" for b in I.B_VALUES]
    ratio = [r1[(False, b)].quantile(0.95) / kv(b) for b in I.B_VALUES]
    rec("单会话 n=50 的 fixed-b 95% 临界值与 Kiefer–Vogelsang 渐近多项式同量级（±15%）",
        all(abs(x - 1) < 0.15 for x in ratio), "; ".join(det_))
    pv = pd.DataFrame(rows)
    pv.to_csv(TAB / "program_validation.csv", index=False)
    th = old_method_theory()
    th.to_csv(TAB / "old_method_theory_cbb_variance_ratio.csv", index=False)
    r6 = th[(th.phi == 0.6) & (th.L.isin([1, 5]))]
    REP.line("旧方法偏宽松的解析解释（AR(1) n=50，去均值）：" + "; ".join(
        f"φ=0.6 L={int(r.L)}: E[CBB 方差]/真实方差={r.expected_cbb_var_over_true_var:.3f}，近似单检验拒绝率 {r.approx_rejection_at_nominal_0_05_single_test:.3f}"
        for r in r6.itertuples()))

    # --- generator truth check
    combos = [(s, n) for s in G.SCENARIOS for n in G.NULL_TYPES]
    tasks = []
    for si, s in enumerate(G.SCENARIOS):
        for ni, n in enumerate(G.NULL_TYPES):
            for c in range(10):
                tasks.append((s, n, [SEED, 5, si, ni, c], 20_000))
    counter.value = 0
    total = len(tasks) * 20_000
    acc = {}
    rt = RateTracker(min_units=N_WORKERS, min_seconds=10)
    last = 0
    REP.line(f"生成模型真值核验：{len(combos)} 个情景×类型 × 200,000 次 = {total:,} 次")
    with make_pool(counter) as pool:
        for k, res in enumerate(pool.imap_unordered(_truth_task, tasks)):
            s, n = res["args"][0], res["args"][1]
            a = acc.setdefault((s, n), dict(sum=0, sumsq=0, n=0, truth=res["truth"]))
            a["sum"] = a["sum"] + res["sum"]; a["sumsq"] = a["sumsq"] + res["sumsq"]; a["n"] += res["n"]
            REP.set_progress(done=counter.value, total=total, unit="次")
            if time.time() - last >= 10 or k + 1 == len(tasks):
                last = time.time()
                REP.line(f"[生成模型与程序验证 / 真值核验] 已完成 {counter.value:,}/{total:,} 次 | 最近：{s}-{n} | "
                         f"{rt.text(k + 1, len(tasks), '块')}")
    tc = []
    for (s, n), a in acc.items():
        mean = a["sum"] / a["n"]
        sd = np.sqrt(a["sumsq"] / a["n"] - mean ** 2)
        se = sd / np.sqrt(a["n"])
        for k, nm in enumerate(I.COMP_NAMES):
            bias = mean[k] - a["truth"][k]
            tc.append(dict(scenario=s, null_type=n, comparison=nm, truth=a["truth"][k], mean_delta_hat=mean[k],
                           bias=bias, mc_se=se[k], sd_delta_hat=sd[k], is_true_null=bool(np.isclose(a["truth"][k], 0)),
                           passed=bool(abs(bias) < 4 * se[k] or abs(bias) < 0.002)))
    tc = pd.DataFrame(tc)
    tc.to_csv(TAB / "generator_truth_check.csv", index=False)
    REP.line(f"真值核验：{int(tc.passed.sum())}/{len(tc)} 个情景×类型×比较的平均 Δ̂ 与真值一致（|偏差|<4 SE 或 <0.002 ms）；"
             f"最大 |偏差| {tc.bias.abs().max():.4f} ms")
    rows.append(dict(check="生成模型真值核验", passed=bool(tc.passed.all()), detail=f"{int(tc.passed.sum())}/{len(tc)}"))
    pd.DataFrame(rows).to_csv(TAB / "program_validation.csv", index=False)
    save_state(program_checks_passed=int(pd.DataFrame(rows).passed.sum()), program_checks_total=len(rows))

    # --- pilot (timing / program only)
    REP.line("小规模试点：每个情景×类型 100 次，全部方法（只检查程序与耗时，不用于方法选择）")
    run_sim_stage("pilot", 0, nsim_ana=100, nsim_boot=100, full=True, label="生成模型与程序验证 / 试点")


# ============================================================================ simulation driver
def run_sim_stage(stage_name, stage_code, nsim_ana, nsim_boot, full, label, groups=("ana", "boot")):
    outdir = CK / "sims" / stage_name
    outdir.mkdir(parents=True, exist_ok=True)
    ch = code_hash()
    tasks = []
    for si, s in enumerate(G.SCENARIOS):
        for ni, n in enumerate(G.NULL_TYPES):
            for gi, grp in enumerate(groups):
                nsim = nsim_ana if grp == "ana" else nsim_boot
                bs = ANA_BATCH if grp == "ana" else BOOT_BATCH
                nb = -(-nsim // bs)
                for b in range(nb):
                    m = min(bs, nsim - b * bs)
                    seed = [SEED, stage_code, si, ni, gi, b]
                    cfg = f"{stage_name}|{s}|{n}|{grp}|{b}|{m}|{BOOT_B}|{ch}|full={full}"
                    import hashlib
                    tasks.append(dict(stage=stage_name, scen=s, ntype=n, group=grp, batch=b, nsim=m, seed=seed,
                                      B=BOOT_B, full=full, cfg_hash=hashlib.sha256(cfg.encode()).hexdigest()[:16],
                                      path=str(outdir / f"{s}_{n}_{grp}_b{b:04d}.npz")))
    done, todo = [], []
    for t in tasks:
        p = Path(t["path"])
        if p.exists():
            try:
                z = np.load(p)
                if str(z["cfg_hash"]) == t["cfg_hash"]:
                    done.append(t)
                    continue
            except Exception:  # noqa: BLE001
                pass
        todo.append(t)
    tot_sims = sum(t["nsim"] for t in tasks)
    done_sims = sum(t["nsim"] for t in done)
    REP.line(f"[{label}] 批次总数 {len(tasks)}（{tot_sims:,} 次模拟）| 断点续跑：已验证可跳过 {len(done)} 批（{done_sims:,} 次）| "
             f"仍需处理 {len(todo)} 批")
    if not todo:
        return
    # long batches first
    todo.sort(key=lambda t: (t["group"] != "boot", t["scen"]))
    counter = mp.get_context("spawn").Value("l", 0)
    n_ok = n_fail = 0
    sims_saved = done_sims
    rt = RateTracker(min_units=N_WORKERS, min_seconds=15)
    last = 0
    manifest = {}
    retry = []
    todo_sims = sum(t["nsim"] for t in todo)
    t_start = time.time()

    def status(last_task):
        el = time.time() - t_start
        live = counter.value
        if live < 1 or el < 15:
            rate = "处理速度：正在估计 | 预计剩余：正在估计"
        else:
            per = el / live
            rate = f"处理速度 {live / el:.1f} 次/秒 | 预计剩余约 {fmt_hms(per * (todo_sims - live))}（预计）"
        REP.line(f"[{label} / 场景 {last_task['scen']} / 类型 {last_task['ntype']} / "
                 f"方法组 {'解析 M1/M3' if last_task['group'] == 'ana' else f'bootstrap M0/M2 B={BOOT_B}'}]\n"
                 f"当前模拟（工作进程实时计数，含进行中批次）{done_sims + live:,}/{tot_sims:,} | 已保存检查点的模拟 {sims_saved:,}\n"
                 f"批次 {n_ok + len(done)}/{len(tasks)} 已保存 | 已运行 {fmt_hms(el)} | {rate}\n"
                 f"失败 {n_fail} | 待重试 {len(retry)}")

    with make_pool(counter) as pool:
        REP.set_task(f"已提交 {len(todo)} 批，等待首批结果")
        it = pool.imap_unordered(_sim_task, todo, chunksize=1)
        while True:
            try:
                res = it.next(timeout=10)
            except mp.TimeoutError:
                REP.set_progress(done=done_sims + counter.value, total=tot_sims, unit="次模拟")
                continue
            except StopIteration:
                break
            t = res["task"]
            if res["ok"]:
                n_ok += 1
                sims_saved += t["nsim"]
                manifest[Path(t["path"]).relative_to(OUT).as_posix()] = dict(
                    cfg_hash=t["cfg_hash"], seed=t["seed"], nsim=t["nsim"], group=t["group"],
                    saved=datetime.now().isoformat(timespec="seconds"), seconds=round(res["seconds"], 2),
                    n_valid_min=res["n_valid_min"])
            else:
                n_fail += 1
                retry.append(t)
                REP.line(f"[{label}] 错误：{t['scen']}-{t['ntype']}-{t['group']} 批次 {t['batch']}。处理：本阶段结束前重试一次。"
                         f"原因：{res['error'][-500:]}")
            REP.set_progress(done=done_sims + counter.value, total=tot_sims, unit="次模拟",
                             extra=f"批次已保存 {n_ok + len(done)}/{len(tasks)}，失败 {n_fail}")
            REP.set_task(f"{t['scen']}-{t['ntype']}-{t['group']} 批次 {t['batch']}")
            REP.event("batch_done", stage=stage_name, scen=t["scen"], ntype=t["ntype"], group=t["group"],
                      batch=t["batch"], ok=res["ok"], seconds=round(res["seconds"], 2))
            if time.time() - last >= 10:
                last = time.time()
                status(t)
                write_manifest(manifest)
                manifest = {}
        if retry:
            REP.line(f"[{label}] 重试 {len(retry)} 个失败批次")
            for res in pool.imap_unordered(_sim_task, retry):
                t = res["task"]
                if res["ok"]:
                    n_ok += 1
                    n_fail -= 1
                    sims_saved += t["nsim"]
                    manifest[Path(t["path"]).relative_to(OUT).as_posix()] = dict(cfg_hash=t["cfg_hash"], seed=t["seed"],
                                                                                 nsim=t["nsim"], retried=True)
                else:
                    REP.line(f"[{label}] 重试仍失败：{t['path']}；原因：{res['error'][-500:]}")
    write_manifest(manifest)
    status(todo[-1])
    if n_fail:
        raise RuntimeError(f"{label}: {n_fail} 个批次重试后仍失败")


def aggregate(stage_name, groups=("ana", "boot")):
    """Sum batch counts -> long table per method x scenario x null type x alpha (+ per comparison)."""
    outdir = CK / "sims" / stage_name
    acc = {}
    full = {}
    for p in sorted(outdir.glob("*.npz")):
        if p.name.endswith(".tmp.npz"):
            continue
        s, n, grp, _ = p.stem.split("_")
        if grp not in groups:
            continue
        z = np.load(p)
        a = acc.setdefault((s, n, grp), {})
        for k in z.files:
            if k.startswith("c_"):
                a[k] = a.get(k, 0) + z[k]
        a["truth"] = z["truth"]
        if "half95" in z.files:
            full.setdefault((s, n, grp), []).append(z["half95"])
    rows, comp_rows = [], []
    for (s, n, grp), a in acc.items():
        names = ANA if grp == "ana" else BOOT
        tn = G.true_null_mask(n)
        nval = a["c_n_valid"]
        med_half = None
        if (s, n, grp) in full:
            hh = np.concatenate(full[(s, n, grp)])
            med_half = np.nanmedian(hh, axis=0)  # (M, 6)
        for mi, m in enumerate(names):
            nv = int(nval[mi])
            for ai, al in enumerate(I.ALPHAS):
                k = int(a["c_fwer"][ai, mi])
                lo, hi = cp_interval(k, nv)
                evaluable = not (grp == "boot" and al == 0.001)
                rows.append(dict(stage=stage_name, method=m, group=grp, scenario=s, null_type=n,
                                 n_true_null=int(tn.sum()), alpha_family=al, n_sims=int(a["c_n"]), n_valid=nv,
                                 n_invalid=int(a["c_n"]) - nv, fwer=k / nv if nv else np.nan, fwer_events=k,
                                 fwer_cp_low=float(lo), fwer_cp_high=float(hi), evaluable=evaluable,
                                 simult_coverage=a["c_covB"][mi] / nv if nv else np.nan,
                                 simult_cov_cp_low=float(cp_interval(a["c_covB"][mi], nv)[0]),
                                 simult_cov_cp_high=float(cp_interval(a["c_covB"][mi], nv)[1])))
            for k in range(6):
                c95 = int(a["c_cov95"][mi, k])
                lo, hi = cp_interval(c95, nv)
                d = dict(stage=stage_name, method=m, group=grp, scenario=s, null_type=n, comparison=I.COMP_NAMES[k],
                         truth=float(a["truth"][k]), is_true_null=bool(tn[k]), n_valid=nv,
                         coverage95=c95 / nv if nv else np.nan, cov95_cp_low=float(lo), cov95_cp_high=float(hi),
                         mean_half95=a["c_sum_half95"][mi, k] / nv if nv else np.nan,
                         median_half95=float(med_half[mi, k]) if med_half is not None else np.nan)
                for ai, al in enumerate(I.ALPHAS):
                    rk = int(a["c_rej"][ai, mi, k])
                    d[f"reject_rate_a{al:g}"] = rk / nv if nv else np.nan
                    lo, hi = cp_interval(rk, nv)
                    d[f"reject_cp_low_a{al:g}"] = float(lo)
                    d[f"reject_cp_high_a{al:g}"] = float(hi)
                comp_rows.append(d)
    return pd.DataFrame(rows), pd.DataFrame(comp_rows)


def verdict_fwer(row):
    al = row["alpha_family"]
    if not row["evaluable"]:
        return "不可评价（重采样分辨率不足）"
    if row["fwer_cp_high"] <= 1.25 * al:
        return "通过"
    if row["fwer_cp_low"] > 1.25 * al:
        return "不通过"
    return "证据不足"


def verdict_cov(lo, hi, target=0.94):
    if lo >= target:
        return "通过"
    if hi < target:
        return "不通过"
    return "证据不足"


# ============================================================================ S5 calibration + selection
def stage_s5():
    run_sim_stage("calibration", 1, nsim_ana=CAL_NSIM, nsim_boot=CAL_NSIM, full=True, label="校准模拟")
    fw, cp = aggregate("calibration")
    fw.to_csv(CAL / "calibration_fwer.csv", index=False)
    cp.to_csv(CAL / "calibration_by_comparison.csv", index=False)
    sel = []
    for m in ALL_METHODS:
        f5 = fw[(fw.method == m) & (fw.alpha_family == 0.05) & fw.scenario.isin(G.CORE)]
        c = cp[(cp.method == m) & cp.scenario.isin(G.CORE)]
        ok_fwer = bool((f5.fwer_cp_high <= 0.075).all())
        ok_cov = bool((c.coverage95 >= 0.93).all())
        cw = cp[(cp.method == m) & (cp.scenario == "C1") & (cp.null_type == "G")]
        sel.append(dict(method=m, analytic=m.startswith(("M1", "M3")), screen_fwer_cp_high_le_0_075=ok_fwer,
                        max_fwer_core=float(f5.fwer.max()), max_fwer_cp_high_core=float(f5.fwer_cp_high.max()),
                        worst_core=f"{f5.loc[f5.fwer.idxmax(), 'scenario']}-{f5.loc[f5.fwer.idxmax(), 'null_type']}",
                        screen_coverage_ge_0_93=ok_cov, min_coverage_core=float(c.coverage95.min()),
                        passes_screen=ok_fwer and ok_cov, C1_mean_median_halfwidth=float(cw.median_half95.mean())))
    sel = pd.DataFrame(sel)
    passing = sel[sel.passes_screen].copy()
    chosen = None
    if len(passing):
        best = passing.C1_mean_median_halfwidth.min()
        tied = passing[passing.C1_mean_median_halfwidth <= best * 1.02].copy()

        def bval(m):
            return float(m.split("_b")[-1]) if "_b" in m else 0.0
        tied["k1"] = ~tied.analytic
        tied["k2"] = -tied.method.map(bval)
        tied = tied.sort_values(["k1", "k2", "C1_mean_median_halfwidth"])
        chosen = tied.iloc[0].method
    sel["selected"] = sel.method == chosen
    sel.to_csv(CAL / "method_selection.csv", index=False)
    (CAL / "method_selection.json").write_text(json.dumps(dict(
        selected=chosen, rule="analysis_plan_calibration_electrodes.md 第 8 节",
        n_passing=int(len(passing)), passing=passing.method.tolist()), indent=2, ensure_ascii=False))
    save_state(selected_method=chosen, n_methods_passing_screen=int(len(passing)))
    REP.line("方法筛选（校准集，核心情景 A/B1/B2/C1/C2 × G/P1/P2，α=0.05）：")
    for r in sel.itertuples():
        REP.line(f"  {r.method:<28} 最大 FWER {r.max_fwer_core:.3f}（CP 上界 {r.max_fwer_cp_high_core:.3f}，{r.worst_core}）"
                 f" | 最小覆盖率 {r.min_coverage_core:.3f} | 通过筛选 {r.passes_screen} | C1 半宽 {r.C1_mean_median_halfwidth:.4f}")
    REP.line(f"按预定规则选出的主要推断方法：{chosen if chosen else '无（没有方法通过筛选）'}")


# ============================================================================ S6 validation
def stage_s6():
    nboot = load_state().get("val_nsim_boot", VAL_NSIM_BOOT)
    run_sim_stage("validation", 2, nsim_ana=VAL_NSIM_ANA, nsim_boot=nboot, full=False, label="独立验证")
    fw, cp = aggregate("validation")
    fw["verdict"] = fw.apply(verdict_fwer, axis=1)
    cp["coverage_verdict"] = [verdict_cov(a, b) for a, b in zip(cp.cov95_cp_low, cp.cov95_cp_high)]
    fw["simult_coverage_verdict"] = [verdict_cov(a, b) for a, b in zip(fw.simult_cov_cp_low, fw.simult_cov_cp_high)]
    fw.to_csv(CAL / "validation_fwer.csv", index=False)
    cp.to_csv(CAL / "validation_by_comparison.csv", index=False)
    summ = []
    for m in ALL_METHODS:
        for al in I.ALPHAS:
            for scope, scen in (("core", G.CORE), ("stress", [s for s in G.SCENARIOS if s not in G.CORE])):
                f = fw[(fw.method == m) & (fw.alpha_family == al) & fw.scenario.isin(scen)]
                v = f.verdict.tolist()
                overall = ("不可评价" if any(x.startswith("不可评价") for x in v) else
                           "通过" if all(x == "通过" for x in v) else "不通过" if any(x == "不通过" for x in v) else "证据不足")
                summ.append(dict(method=m, alpha_family=al, scope=scope, verdict=overall,
                                 n_pass=v.count("通过"), n_fail=v.count("不通过"), n_insufficient=v.count("证据不足"),
                                 max_fwer=float(f.fwer.max()), failing=";".join(f"{r.scenario}-{r.null_type}" for r in f.itertuples()
                                                                                if r.verdict == "不通过")))
        c = cp[(cp.method == m) & cp.scenario.isin(G.CORE)]
        cv = c.coverage_verdict.tolist()
        summ.append(dict(method=m, alpha_family=np.nan, scope="core_coverage95",
                         verdict="通过" if all(x == "通过" for x in cv) else "不通过" if any(x == "不通过" for x in cv) else "证据不足",
                         n_pass=cv.count("通过"), n_fail=cv.count("不通过"), n_insufficient=cv.count("证据不足"),
                         max_fwer=float(c.coverage95.min()), failing=""))
    summ = pd.DataFrame(summ)
    summ.to_csv(CAL / "validation_verdicts_by_method.csv", index=False)
    chosen = load_state().get("selected_method")
    if chosen:
        s = summ[summ.method == chosen]
        REP.line(f"主要方法 {chosen} 的独立验证判定：")
        for r in s.itertuples():
            REP.line(f"  {r.scope:<16} α={r.alpha_family} → {r.verdict}（通过 {r.n_pass}，不通过 {r.n_fail}，证据不足 {r.n_insufficient}）"
                     + (f" 失败：{r.failing}" if r.failing else ""))
        save_state(selected_validation={f"{r.scope}_{r.alpha_family}": r.verdict for r in s.itertuples()})
    for m in ["M0_CBB_abs_L1", "M0_CBB_abs_L5"]:
        f = fw[(fw.method == m) & (fw.alpha_family == 0.05) & (fw.null_type == "G")]
        REP.line(f"旧方法 {m}（α=0.05，全局零假设）FWER：" + "; ".join(f"{r.scenario} {r.fwer:.3f}" for r in f.itertuples()))


# ============================================================================ S7 real data
SCOPES = {"pooled_3_sessions": [0, 1, 2], "ses-01_only": [0], "ses-02_only": [1], "ses-03_only": [2],
          "drop_ses-01": [1, 2], "drop_ses-02": [0, 2], "drop_ses-03": [0, 1]}
VARIANT_TEXT = {
    "main_pooled_nonzero": ("主要对象", "池化均值 ΣS/ΣN，非零窗口，全部试次"),
    "A_including_zero": ("敏感性 A", "池化均值，包含有限零延迟窗口"),
    "B_artifact_excluded": ("敏感性 B", "池化均值，非零窗口，NWB 伪迹试次权重 0（保留试次网格）"),
    "D_pair_equal": ("敏感性 D（不同加权）", "通道对内均值 → 通道对等权 → 试次等权（无缺失时等于会话等权）"),
    "E_trial_median": ("敏感性 E（不同统计量）", "试次×脑区中位数 → 试次等权均值之差（不是池化中位数之差）"),
}


def real_arrays():
    trs, pts, meta = load_trial_tables()
    key = meta.session + "_" + meta.trial_pos.astype(str)
    order = {k: i for i, k in enumerate(key)}
    T = len(meta)
    S = np.zeros((T, 4)); N = np.zeros((T, 4)); Nall = np.zeros((T, 4)); MED = np.full((T, 4), np.nan)
    for r in trs.itertuples():
        t = order[f"{r.session}_{r.trial_pos}"]; j = I.REGIONS.index(r.region)
        S[t, j], N[t, j], Nall[t, j] = r.S_nonzero_ms, r.N_nonzero, r.n_windows_raw
        if r.N_nonzero > 0:
            MED[t, j] = r.median_nonzero_ms
    V = np.full((T, 4), np.nan)
    pts = pts.copy()
    pts["t"] = [order[f"{s}_{p}"] for s, p in zip(pts.session, pts.trial_pos)]
    for (t, r), g in pts.groupby(["t", "region"]):
        m = g.mean_nonzero_ms.to_numpy()
        m = m[np.isfinite(m)]
        if m.size:
            V[t, I.REGIONS.index(r)] = m.mean()
    art = meta.artifact.astype(bool).to_numpy()
    return meta, pts, S, N, Nall, MED, V, art


def variant_inputs(name, S, N, Nall, MED, V, art):
    T = S.shape[0]
    ones = np.ones(T)
    if name == "main_pooled_nonzero":
        return S, N, ones
    if name == "A_including_zero":
        return S, Nall, ones
    if name == "B_artifact_excluded":
        return S, N, (~art).astype(float)
    if name == "D_pair_equal":
        return np.nan_to_num(V), np.isfinite(V).astype(float), ones
    if name == "E_trial_median":
        return np.nan_to_num(MED), np.isfinite(MED).astype(float), ones
    raise KeyError(name)


def subset(S, N, w, ses_idx):
    rows = np.concatenate([np.arange(i * 50, (i + 1) * 50) for i in ses_idx])
    sessions = [(k * 50, (k + 1) * 50) for k in range(len(ses_idx))]
    return S[rows], N[rows], w[rows], sessions


def analyse(S, N, w, sessions, boot_seed=None, B=REAL_B, Ls=I.LS):
    ns = len(sessions)
    a = I.analytic_inference(S[None], N[None], w, sessions, load_refs(ns))
    out = {}
    for mi, m in enumerate(ANA):
        out[m] = dict(delta=a["delta"][0], p=a["p"][0, mi], half95=a["half95"][0, mi], halfB=a["halfB"][0, mi],
                      se=a["se"][0, mi], n_valid=np.full(6, np.nan), B=np.nan)
    if boot_seed is not None:
        rng = np.random.default_rng(np.random.SeedSequence(boot_seed))
        bo = I.boot_inference(S, N, w, sessions, rng, B, Ls=Ls, chunk=5000)
        names = [f"M0_CBB_abs_L{L}" for L in Ls] + [f"M2_CBB_studentized_L{L}" for L in Ls]
        for mi, m in enumerate(names):
            out[m] = dict(delta=bo["delta"], p=bo["p"][mi], half95=bo["half95"][mi], halfB=bo["halfB"][mi],
                          se=np.full(6, np.nan), n_valid=bo["n_valid"][mi], B=B)
    return out


def method_param(m):
    if "_b" in m:
        return f"Bartlett b={m.split('_b')[-1]}（M={int(round(float(m.split('_b')[-1]) * 50))} 试次/会话）"
    return f"块长 L={m.split('_L')[-1]} 试次"


def real_diagnostics(S, N, w, sessions):
    u, d = I.influence(S, N, w)
    r = I._residualize(u, sessions, w, False)
    rows = []
    for k, nm in enumerate(I.COMP_NAMES):
        num = den = 0.0
        tvals = []
        for s0, s1 in sessions:
            x = r[s0:s1, k]
            num += (x[1:] * x[:-1]).sum(); den += (x * x).sum()
            t = np.arange(s1 - s0) - (s1 - s0 - 1) / 2
            beta = (x * t).sum() / (t * t).sum()
            res = x - beta * t
            se = np.sqrt((res ** 2).sum() / (len(x) - 2) / (t * t).sum())
            tvals.append(beta / se if se > 0 else 0.0)
        rows.append(dict(comparison=nm, lag1_acf_influence=num / den, max_abs_trend_t=float(np.max(np.abs(tvals))),
                         trend_t_by_session=";".join(f"{v:+.2f}" for v in tvals)))
    return pd.DataFrame(rows)


def stage_s7():
    st = load_state()
    chosen = st.get("selected_method")
    vv = pd.read_csv(CAL / "validation_verdicts_by_method.csv")
    fwv = pd.read_csv(CAL / "validation_fwer.csv")

    def vstr(m):
        s = vv[(vv.method == m) & (vv.scope == "core")]
        return "; ".join(f"α={r.alpha_family:g}:{r.verdict}" for r in s.itertuples())

    def passed(m, al, scope="core"):
        s = vv[(vv.method == m) & (vv.scope == scope) & (vv.alpha_family == al)]
        return len(s) == 1 and s.verdict.iloc[0] == "通过"

    # validated dependence range for the selected method (all null types must pass at alpha=0.05)
    phi_of = {"A": 0.0, "B1": 0.3, "B2": 0.6, "D1": 0.85}
    phi_ceiling, trend_ok = None, None
    if chosen:
        f = fwv[(fwv.method == chosen) & (fwv.alpha_family == 0.05)]
        okscen = {s: (f[f.scenario == s].verdict == "通过").all() for s in G.SCENARIOS}
        phi_ceiling = 0.0 if okscen["A"] else None
        for s in ("B1", "B2", "D1"):
            if phi_ceiling is not None and okscen[s]:
                phi_ceiling = phi_of[s]
            else:
                break
        trend_ok = bool(okscen["D3"])
    meta, pts, S, N, Nall, MED, V, art = real_arrays()
    REP.line(f"主要方法：{chosen}；已验证的 lag-1 自相关上限 {phi_ceiling}；趋势情景 D3 通过：{trend_ok}")
    rows = []
    diag_rows = []
    jobs = [(v, sc) for v in VARIANT_TEXT for sc in SCOPES]
    REP.set_progress(done=0, total=len(jobs), unit="数据集")
    rt = RateTracker(min_units=2, min_seconds=5)
    for j, (v, sc) in enumerate(jobs):
        REP.set_task(f"{v} / {sc}：解析方法 + bootstrap（B={REAL_B}，L=1,2,5,10）")
        Sv, Nv, wv = variant_inputs(v, S, N, Nall, MED, V, art)
        Ss, Ns, ws, sessions = subset(Sv, Nv, wv, SCOPES[sc])
        vi = list(VARIANT_TEXT).index(v); si = list(SCOPES).index(sc)
        res = analyse(Ss, Ns, ws, sessions, boot_seed=[SEED, 3, vi, si])
        dg = real_diagnostics(Ss, Ns, ws, sessions)
        dg["analysis_variant"] = v; dg["session_scope"] = sc
        diag_rows.append(dg)
        for m, o in res.items():
            for k, (a_, b_) in enumerate(I.COMPARISONS):
                rows.append(row_for(m, o, k, v, sc, "原始映射", "全部 60 个区内通道对", int(ws.sum()),
                                    vstr(m), chosen, passed, dg.iloc[k], phi_ceiling, trend_ok))
        REP.set_progress(done=j + 1)
        REP.line(f"[真实数据重新检验] 数据集 {j + 1}/{len(jobs)}：{v} / {sc} 完成 | {rt.text(j + 1, len(jobs), '数据集')}")
    # electrode-handling data sets (main estimand, pooled scope)
    pc = pd.read_csv(ELE / "pair_classification.csv")
    excl = set(st.get("excluded_pairs", []))
    ex_idx = {int(r.pair_index) for r in pc.itertuples() if f"{r.label_i}–{r.label_j}" in excl}
    contacts = sorted(set(pc.label_i) | set(pc.label_j), key=lambda s: (bundle_of(s), int(s[len(bundle_of(s)):])))
    sets = [("证据支持的纠正", f"排除满足预定数据质量规则的通道对（{len(ex_idx)} 对）", lambda r: r.pair_index not in ex_idx, False),
            ("未确认敏感性", "排除零延迟比例 ≥70% 的通道对", lambda r: r.zero_fraction_all_trials < 0.70, False),
            ("未确认敏感性", "只用同侧标签通道对", lambda r: bool(r.same_side_label), True),
            ("未确认敏感性", "只用异侧标签通道对", lambda r: not r.same_side_label, True),
            ("未确认敏感性", "只用同束（同一电极）通道对", lambda r: bool(r.same_bundle), True),
            ("未确认敏感性", "只用跨束通道对", lambda r: not r.same_bundle, True),
            ("未确认敏感性", "排除涉及 ECL 的通道对（侧别冲突）", lambda r: not r.involves_ECL, True)]
    for c in contacts:
        sets.append(("未确认敏感性", f"去一触点：{c}", (lambda cc: (lambda r: cc not in (r.label_i, r.label_j)))(c), False))
    REP.set_progress(done=0, total=len(sets), unit="电极处理数据集")
    elec_rows = []
    pts2 = pts.copy()
    order = {f"{s}_{p}": i for i, (s, p) in enumerate(zip(meta.session, meta.trial_pos))}
    pts2["t"] = [order[f"{s}_{p}"] for s, p in zip(pts2.session, pts2.trial_pos)]
    for j, (ver, desc, keep_fn, strat) in enumerate(sets):
        keep = {int(r.pair_index) for r in pc.itertuples() if keep_fn(r)}
        sub = pts2[pts2.pair_index.isin(keep)]
        Se = np.zeros((150, 4)); Ne = np.zeros((150, 4))
        for (t, r), g in sub.groupby(["t", "region"]):
            Se[t, I.REGIONS.index(r)] = g.S_nonzero_ms.sum(); Ne[t, I.REGIONS.index(r)] = g.N_nonzero.sum()
        npairs = {r: int(((pc.region == r) & pc.pair_index.isin(keep)).sum()) for r in I.REGIONS}
        ncont = {r: len(set(pc[(pc.region == r) & pc.pair_index.isin(keep)].label_i) |
                        set(pc[(pc.region == r) & pc.pair_index.isin(keep)].label_j)) for r in I.REGIONS}
        removed = {r: int((pc.region == r).sum()) - npairs[r] for r in I.REGIONS}
        REP.set_task(f"{desc}")
        res = analyse(Se, Ne, np.ones(150), G.SESSIONS, boot_seed=[SEED, 3, 100 + j, 0], B=ELEC_B, Ls=[5])
        dg = real_diagnostics(Se, Ne, np.ones(150), G.SESSIONS)
        for m, o in res.items():
            for k, (a_, b_) in enumerate(I.COMPARISONS):
                ra, rb = I.REGIONS[a_], I.REGIONS[b_]
                row = row_for(m, o, k, "main_pooled_nonzero", "pooled_3_sessions", ver, desc, 150, vstr(m), chosen,
                              passed, dg.iloc[k], phi_ceiling, trend_ok)
                row.update(pairs_A=npairs[ra], pairs_B=npairs[rb], contacts_A=ncont[ra], contacts_B=ncont[rb],
                           pairs_removed_A=removed[ra], pairs_removed_B=removed[rb])
                if strat and min(npairs[ra], npairs[rb]) < 3:
                    row["decision"] = "仅描述（某一脑区通道对 < 3，不做检验）" if m == chosen else row["decision"]
                    row["test_performed"] = False
                if min(npairs[ra], npairs[rb]) == 0:
                    row["decision"] = "无数据（某一脑区无剩余通道对）"
                    row["test_performed"] = False
                elec_rows.append(row)
        REP.set_progress(done=j + 1)
        if (j + 1) % 5 == 0 or j + 1 == len(sets):
            REP.line(f"[真实数据重新检验 / 电极处理] {j + 1}/{len(sets)}：{desc}")
    allr = pd.DataFrame(rows + elec_rows)
    allr.to_csv(REAL / "results_all_methods_all_variants.csv", index=False)
    pd.concat(diag_rows).to_csv(REAL / "diagnostics_influence_acf_trend.csv", index=False)
    if chosen:
        sel = allr[allr.method == chosen]
        sel.to_csv(REAL / "results_selected_method.csv", index=False)
        main = sel[(sel.analysis_variant == "main_pooled_nonzero") & (sel.session_scope == "pooled_3_sessions") &
                   (sel.data_version == "原始映射")]
        for r in main.itertuples():
            REP.line(f"  主要结果 {r.region_A_short}−{r.region_B_short}: Δ={r.difference_ms:+.3f} ms，95% CI [{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]，"
                     f"p_adj={fmt_p(r.adjusted_p)} → {r.decision}")
    save_state(real_rows=len(allr))


def row_for(m, o, k, variant, scope, data_version, electrode_handling, n_trials_eff, vstring, chosen, passed, dg,
            phi_ceiling, trend_ok):
    a_, b_ = I.COMPARISONS[k]
    p = float(o["p"][k])
    padj = min(6 * p, 1.0) if np.isfinite(p) else np.nan
    d = float(o["delta"][k])
    dec_ = ""
    lim = []
    if m == chosen:
        if not np.isfinite(padj):
            dec_ = "无法计算"
        else:
            lvl = [al for al in (0.001, 0.01, 0.05) if passed(m, al) and padj < al]
            if lvl:
                dec_ = f"显著（p_adj<{lvl[0]:g}，该水平已通过独立验证）"
            elif padj < 0.05:
                dec_ = "证据不足（p_adj<0.05，但方法未在该水平通过独立验证）"
            elif passed(m, 0.05):
                dec_ = "不显著（不等于无差异）"
            else:
                dec_ = "证据不足"
            if phi_ceiling is not None and dg.lag1_acf_influence > phi_ceiling + 1e-9:
                lim.append(f"lag-1 自相关 {dg.lag1_acf_influence:+.2f} 超过已验证上限 {phi_ceiling}")
            if trend_ok is False and dg.max_abs_trend_t > 2:
                lim.append(f"会话内趋势 |t|={dg.max_abs_trend_t:.1f}，趋势情景未通过验证")
            if lim and dec_.startswith("显著"):
                dec_ = "证据不足（真实数据诊断超出已验证范围）"
    elif chosen is None:
        dec_ = "证据不足（没有方法通过校准筛选）"
    return dict(data_version=data_version, electrode_handling=electrode_handling, analysis_variant=variant,
                analysis_role=VARIANT_TEXT[variant][0], estimand=VARIANT_TEXT[variant][1], session_scope=scope,
                region_A=I.REGIONS[a_], region_B=I.REGIONS[b_], region_A_short=I.SHORT[I.REGIONS[a_]],
                region_B_short=I.SHORT[I.REGIONS[b_]], direction=f"{I.SHORT[I.REGIONS[a_]]} − {I.SHORT[I.REGIONS[b_]]}",
                difference_ms=d, ci95_low=d - o["half95"][k], ci95_high=d + o["half95"][k], ci95_level=0.95,
                ci_bonf_low=d - o["halfB"][k], ci_bonf_high=d + o["halfB"][k], ci_bonf_level=1 - 0.05 / 6,
                raw_p=p, adjusted_p=padj, method=m, method_param=method_param(m),
                bootstrap_B=o["B"], n_valid_resamples=o["n_valid"][k], n_trials_effective=n_trials_eff,
                is_selected_method=(m == chosen), calibration_scope=vstring, decision=dec_,
                diagnostics=f"lag-1 ACF(影响值)={dg.lag1_acf_influence:+.2f}; 会话内趋势 t={dg.trend_t_by_session}",
                limitations="; ".join(lim), test_performed=True)


# ============================================================================ S8
def stage_s8():
    import report_numbers as RN
    RN.write(REP)
    import write_reports as WR
    WR.write(REP)
    import figures as F
    F.make_all(REP)
    expected = [
        "analysis_plan_calibration_electrodes.md", "deviations_from_plan.md",
        "tables/provenance_nwb_sha256.csv", "tables/previous_round_errata.csv", "tables/program_validation.csv",
        "tables/generator_truth_check.csv", "tables/fixedb_reference_quantiles.csv",
        "tables/old_method_theory_cbb_variance_ratio.csv",
        "electrodes/channel_verification_table.csv", "electrodes/electrode_findings.csv",
        "electrodes/pair_classification.csv", "electrodes/questions_for_authors.md", "electrodes/sources.md",
        "waveform_qc/pair_metrics_summary.csv", "waveform_qc/micro_best_macro_match.csv",
        "calibration/calibration_fwer.csv", "calibration/method_selection.csv", "calibration/validation_fwer.csv",
        "calibration/validation_by_comparison.csv", "calibration/validation_verdicts_by_method.csv",
        "realdata/results_all_methods_all_variants.csv", "realdata/old_vs_new_significance.csv",
        "reports/results_numeric_summary_CN.md",
        "REPORT_CN.md", "methods_paragraph_EN.md", "response_reviewer_2_4_2_EN.md",
        "run.log", "progress.jsonl", "timing_summary.csv", "run_state.json", "checkpoint_manifest.json",
        "logs/commands.txt", "logs/environment.json", "logs/time_estimate_at_start.json",
    ]
    for stem in ["fig1_electrode_metadata_conflicts", "fig2_waveform_qc_pairs", "fig3_micro_macro_correlation",
                 "fig4_calibration_fwer", "fig5_validation_fwer_coverage", "fig6_differences_ci_selected",
                 "fig7_sensitivity_electrode_session_weight", "fig8_old_vs_new"]:
        expected += [f"figures/{stem}.png", f"figures/{stem}.pdf"]
    missing = [e for e in expected if not (OUT / e).exists()]
    empty = [e for e in expected if (OUT / e).exists() and (OUT / e).stat().st_size == 0]
    man = json.loads((OUT / "checkpoint_manifest.json").read_text()) if (OUT / "checkpoint_manifest.json").exists() else {}
    n_ck = len(man)
    n_missing_ck = sum(1 for k in man if not (OUT / k).exists())
    fw = pd.read_csv(CAL / "validation_fwer.csv") if (CAL / "validation_fwer.csv").exists() else pd.DataFrame()
    exp_rows = len(ALL_METHODS) * len(G.SCENARIOS) * len(G.NULL_TYPES) * len(I.ALPHAS)
    integ = dict(expected_files=len(expected), missing=missing, empty=empty, checkpoints_in_manifest=n_ck,
                 checkpoints_missing=n_missing_ck, validation_rows=len(fw), validation_rows_expected=exp_rows)
    (OUT / "integrity_check.json").write_text(json.dumps(integ, indent=2, ensure_ascii=False))
    REP.line(f"完整性检查：预期文件 {len(expected)}，缺失 {len(missing)} {missing[:8]}，空文件 {len(empty)}；"
             f"检查点登记 {n_ck}，缺失 {n_missing_ck}；验证结果行 {len(fw)}/{exp_rows}")
    return not missing and not empty and n_missing_ck == 0 and len(fw) == exp_rows


# ============================================================================ main
STAGE_FN = {"s1": stage_s1, "s2": stage_s2, "s3": stage_s3, "s4": stage_s4, "s5": stage_s5, "s6": stage_s6,
            "s7": stage_s7, "s8": stage_s8}


def main():
    global REP
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default=",".join(k for k, _ in STAGES))
    args = ap.parse_args()
    want = [s.strip() for s in args.stages.split(",") if s.strip()]
    REP = Reporter(OUT, heartbeat_s=10)
    t_run = time.time()
    REP.line(f"Review 2.4.2 / 图 4c 校准与电极核验流水线启动 | PID {os.getpid()} | 阶段 {want} | 输出 {OUT}")
    REP.event("run_start", stages=want)
    names = dict(STAGES)
    done_stages = []
    complete = False
    for s in want:
        REP.start_stage(names[s])
        try:
            r = STAGE_FN[s]()
            if s == "s8" and r is False:
                REP.end_stage("完成（发现未完成事项）")
                done_stages.append(s)
                continue
            REP.end_stage("完成")
            done_stages.append(s)
        except Exception as e:  # noqa: BLE001
            REP.line(f"[{names[s]}] 错误：{type(e).__name__}: {e}")
            REP.line(traceback.format_exc())
            REP.line(f"处理：停止本阶段及后续阶段（{[names[x] for x in want[want.index(s) + 1:]]}），修复后可用 --stages 从此处续跑。")
            REP.end_stage("失败", note=str(e)[:200])
            break
        complete = s == "s8" and r is True
    el = time.time() - t_run
    own, kids = peak_rss_mb()
    REP.line("=" * 78)
    if complete:
        REP.line("全部完成")
    else:
        left = [names[x] for x in want if x not in done_stages]
        REP.line(f"本次运行结束（未全部完成）。本次完成阶段：{[names[x] for x in done_stages]}；未完成：{left or '无（但完整性检查发现缺项或未运行 s8）'}")
    REP.line(f"本次运行耗时 {fmt_hms(el)} | 峰值内存 主进程 {own:.0f} MB / 单个子进程最大 {kids:.0f} MB | 输出目录 {OUT}")
    if (OUT / "timing_summary.csv").exists():
        ts = pd.read_csv(OUT / "timing_summary.csv")
        for r in ts.itertuples():
            REP.line(f"  {r.stage:<16} {r.status:<10} {float(r.seconds):>9.1f} s")
    REP.line(f"主报告：{OUT / 'REPORT_CN.md'}；主要图表：{FIG}")
    REP.event("run_end", complete=complete, seconds=round(el, 1))
    REP.close()


if __name__ == "__main__":
    main()
