"""Figures for the calibration / electrode round. Each figure is saved as PNG and PDF with its source data."""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import genmodels as G  # noqa: E402
import infer as I  # noqa: E402
from common import CAL, ELE, FIG, REAL, WQC, load_state  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "figure.dpi": 150,
                     "font.sans-serif": ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"],
                     "axes.unicode_minus": False})
SRC = FIG / "source_data"
REG_COL = {"Amygdala": "#1b9e77", "Hippocampus": "#d95f02", "Middle Temporal Gyrus": "#7570b3",
           "Superior Temporal Gyrus": "#e7298a", "unspecific": "#cccccc", "Insular Gyrus": "#66a61e",
           "Inferior Temporal Gyrus": "#e6ab02"}


def display():
    """Selected method if any; otherwise the calibration-chosen display methods (bootstrap, analytic)."""
    chosen = load_state().get("selected_method")
    if chosen:
        return [chosen], chosen
    d = json.loads((CAL / "display_methods.json").read_text())
    return [d["bootstrap"], d["analytic"]], d["analytic"]


def save(fig, stem, REP=None):
    fig.savefig(FIG / f"{stem}.png", dpi=200, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    if REP:
        REP.line(f"  图：figures/{stem}.png/.pdf")


def fig1(REP):
    mac = pd.read_csv(ELE / "macro_contacts_ses-01.csv")
    ch = pd.read_csv(ELE / "channel_verification_table.csv")
    c1 = ch[ch.session == "ses-01"]
    micro_rows = set(c1.electrode_table_row)
    mac["used_by_micro_series"] = mac.electrode_table_row.isin(micro_rows)
    mac.to_csv(SRC / "fig1_macro_contacts.csv", index=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw=dict(width_ratios=[1.1, 1]))
    a = ax[0]
    for bnd, g in mac.groupby("bundle"):
        g = g.sort_values("label", key=lambda c: c.str.extract(r"(\d+)$")[0].astype(int))
        a.plot(g.mni_x_inferred, g.mni_y_inferred, "-", color="0.7", lw=0.8, zorder=1)
        for r in g.itertuples():
            col = "#2166ac" if r.lateral_label == "Left" else "#b2182b" if r.lateral_label == "Right" else "0.6"
            a.scatter(r.mni_x_inferred, r.mni_y_inferred, s=28 if r.used_by_micro_series else 12, c=col,
                      edgecolors="k" if r.used_by_micro_series else "none", linewidths=0.6, zorder=2)
        t = g.iloc[0]
        a.annotate(bnd, (t.mni_x_inferred, t.mni_y_inferred), xytext=(3, 3), textcoords="offset points", fontsize=8,
                   fontweight="bold")
    a.axvline(0, color="k", lw=0.6, ls="--")
    a.set_xlabel("推断 MNI x = NWB z（+右）")
    a.set_ylabel("推断 MNI y = −NWB x（+前）")
    a.set_title("A  宏触点位置（颜色=location 字符串侧别：蓝左/红右）\n黑边=同一电极表行也被微电极序列引用")
    a.text(0.02, 0.02, "ECL：字符串写 Right，坐标在左侧\nLR：字符串写 Left，坐标在右侧", transform=a.transAxes, fontsize=7,
           va="bottom", bbox=dict(fc="w", ec="0.6"))
    b = ax[1]
    bundles = ["AHL", "AL", "ECL", "PHR"]
    regs = sorted(set(c1.region))
    code = {r: i for i, r in enumerate(regs)}
    M = np.zeros((4, 8))
    for i, bnd in enumerate(bundles):
        for j in range(8):
            r = c1[c1.label == f"{bnd}{j + 1}"].region.iloc[0]
            M[i, j] = code[r]
            short = {"Amygdala": "AMY", "Hippocampus": "HIP", "Middle Temporal Gyrus": "MTG",
                     "Superior Temporal Gyrus": "STG", "unspecific": "—", "Insular Gyrus": "INS",
                     "Inferior Temporal Gyrus": "ITG"}[r]
            b.text(j, i, short, ha="center", va="center", fontsize=7)
    from matplotlib.colors import ListedColormap
    b.imshow(M, cmap=ListedColormap([REG_COL[r] for r in regs]), aspect="auto", alpha=0.75)
    b.set_yticks(range(4), bundles)
    b.set_xticks(range(8), [f"微电极 {k}" for k in range(1, 9)], rotation=45)
    b.set_title("B  32 个微电极通道被赋予的脑区（来自同编号宏触点）\n数据描述：同一电极的微电极均从尖端伸出约 4 mm")
    fig.tight_layout()
    save(fig, "fig1_electrode_metadata_conflicts", REP)


def fig2(REP):
    tr = np.load(WQC / "example_traces_ses-01_trial00_4s_20ms.npz")
    summ = pd.read_csv(WQC / "pair_metrics_summary.csv")
    order = summ.sort_values(["selection_reason", "pair"]).pair.tolist()
    show = [p for p in ["AL1–AL2", "PHR6–PHR7", "ECL5–ECL6", "AHL1–AHL2", "AHL1–PHR1", "AL2–ECL1"] if p in order]
    fig = plt.figure(figsize=(10, 6.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.3])
    t = np.arange(640) / 32.0
    for k, p in enumerate(show):
        a = fig.add_subplot(gs[k // 3, k % 3])
        key = p.replace("–", "-")
        x = tr[key]
        a.plot(t, x[0], lw=0.6, label=p.split("–")[0])
        a.plot(t, x[1], lw=0.6, label=p.split("–")[1])
        r = summ[summ.pair == p].iloc[0]
        a.set_title(f"{p}（{r.selection_reason}）\n零时差 r={r.r_zero_lag_median:.2f}，零延迟窗口 {r.window_zero_fraction_mean:.0%}",
                    fontsize=7)
        a.set_xlabel("ms（ses-01 trial 0，第 4 s 起 20 ms）")
        a.set_ylabel("数据单位（声明 V，疑似 µV）", fontsize=6)
        a.legend(fontsize=6, loc="upper right")
    a = fig.add_subplot(gs[2, :])
    s2 = summ.set_index("pair").loc[order]
    xx = np.arange(len(order))
    a.bar(xx - 0.27, s2.r_zero_lag_median, 0.27, label="零时差相关（中位数，15 试次）")
    a.bar(xx, s2.window_zero_fraction_mean, 0.27, label="零延迟窗口比例（15 试次）")
    a.bar(xx + 0.27, s2.diff_rms_ratio_median / 2, 0.27, label="差值 RMS / 平均 RMS ÷ 2")
    a.set_xticks(xx, [f"{p}\n{s2.loc[p, 'selection_reason']}" for p in order], rotation=60, fontsize=6)
    a.axhline(0, color="k", lw=0.5)
    a.legend(fontsize=7, ncol=3, loc="upper left")
    a.set_title("预定 12 个通道对的波形指标；逐样本相同比例均 ≤0.001，无通道满足预定排除规则")
    summ.to_csv(SRC / "fig2_pair_metrics_summary.csv", index=False)
    fig.tight_layout()
    save(fig, "fig2_waveform_qc_pairs", REP)


def fig3(REP):
    mm = pd.read_csv(WQC / "micro_vs_macro_correlation_lowpass200Hz.csv", index_col=0)
    uu = pd.read_csv(WQC / "micro_vs_micro_correlation_lowpass200Hz.csv", index_col=0)
    fig, ax = plt.subplots(1, 5, figsize=(13, 3.2), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.6]))
    for i, bnd in enumerate(["AHL", "AL", "ECL", "PHR"]):
        rows = [f"{bnd}{k}" for k in range(1, 9)]
        sub = mm.loc[rows, rows]
        im = ax[i].imshow(sub.to_numpy(), vmin=-0.5, vmax=0.5, cmap="RdBu_r")
        ax[i].set_xticks(range(8), [c[len(bnd):] for c in rows])
        ax[i].set_yticks(range(8), [c[len(bnd):] for c in rows])
        ax[i].set_xlabel(f"{bnd} 宏触点")
        ax[i].set_ylabel(f"{bnd} 微电极")
        ax[i].set_title(f"{bnd}：微电极 × 宏触点 r")
    im2 = ax[4].imshow(uu.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
    ax[4].set_xticks(range(0, 32, 8), ["AHL", "AL", "ECL", "PHR"])
    ax[4].set_yticks(range(0, 32, 8), ["AHL", "AL", "ECL", "PHR"])
    ax[4].set_title("微电极 × 微电极 r")
    fig.colorbar(im, ax=ax[:4], shrink=0.7, label="r（低通 200 Hz，15 试次 Fisher-z 平均）")
    fig.colorbar(im2, ax=ax[4], shrink=0.7)
    fig.suptitle("微电极与宏电极、微电极之间的零时差相关（预定描述性检查）", y=1.02)
    save(fig, "fig3_micro_macro_correlation", REP)


def _short(m):
    return (m.replace("M0_CBB_abs_", "M0 ").replace("M2_CBB_studentized_", "M2 ")
            .replace("M1_HAC_fixedb_", "M1 ").replace("M3_HACdetrend_fixedb_", "M3 "))


def fig4(REP):
    fw = pd.read_csv(CAL / "calibration_fwer.csv")
    f = fw[fw.alpha_family == 0.05]
    meths = I.boot_method_names() + I.analytic_method_names()
    scen = list(G.SCENARIOS)
    fig, ax = plt.subplots(1, 3, figsize=(13, 5.2), sharey=True)
    for i, n in enumerate(G.NULL_TYPES):
        M = f[f.null_type == n].pivot_table(index="method", columns="scenario", values="fwer").loc[meths, scen]
        im = ax[i].imshow(M.to_numpy(), cmap="RdYlGn_r", vmin=0, vmax=0.25, aspect="auto")
        for (r, c), v in np.ndenumerate(M.to_numpy()):
            ax[i].text(c, r, f"{v:.3f}", ha="center", va="center", fontsize=5.5,
                       color="k" if v < 0.18 else "w")
        ax[i].set_xticks(range(len(scen)), scen)
        ax[i].set_yticks(range(len(meths)), [_short(m) for m in meths])
        ax[i].axvline(4.5, color="k", lw=1)
        ax[i].set_title(f"{n}：{G.NULL_TYPES[n][0]}")
    fig.colorbar(im, ax=ax, shrink=0.6, label="家族错误率（名义 0.05）")
    fig.suptitle(f"校准集（每格 {int(f.n_valid.max())} 次模拟）：家族 α=0.05 下的 FWER；竖线左侧为核心情景，右侧为压力情景", y=1.0)
    f.to_csv(SRC / "fig4_calibration_fwer_alpha0.05.csv", index=False)
    save(fig, "fig4_calibration_fwer", REP)


def fig5(REP):
    fw = pd.read_csv(CAL / "validation_fwer.csv")
    cp = pd.read_csv(CAL / "validation_by_comparison.csv")
    chosen = load_state().get("selected_method")
    disp, _ = display()
    meths = list(dict.fromkeys(disp + ["M0_CBB_abs_L5", "M0_CBB_abs_L1"]))
    scen = list(G.SCENARIOS)
    fig, ax = plt.subplots(2, 3, figsize=(13, 7))
    for j, al in enumerate(I.ALPHAS):
        a = ax[0, j]
        for k, m in enumerate(meths):
            f = fw[(fw.method == m) & (fw.alpha_family == al)]
            for ni, n in enumerate(G.NULL_TYPES):
                g = f[f.null_type == n].set_index("scenario").loc[scen]
                x = np.arange(len(scen)) + (k - 1.5) * 0.2 + (ni - 1) * 0.05
                ok = g.evaluable.to_numpy().astype(bool)
                a.errorbar(x[ok], g.fwer[ok], yerr=[(g.fwer - g.fwer_cp_low)[ok], (g.fwer_cp_high - g.fwer)[ok]], fmt="o",
                           ms=2.5, lw=0.7, color=f"C{k}", label=_short(m) if ni == 0 else None)
        a.axhline(al, color="k", lw=0.8)
        a.axhline(1.25 * al, color="k", lw=0.6, ls="--")
        a.set_yscale("log")
        a.set_xticks(range(len(scen)), scen)
        a.axvline(4.5, color="0.5", lw=0.6)
        a.set_title(f"独立验证：家族 α={al:g}（实线名义，虚线 1.25α）")
        a.set_ylabel("FWER（95% CP 区间）")
        if j == 0:
            a.legend(fontsize=7)
    for k, m in enumerate(meths):
        a = ax[1, 0]
        c = cp[(cp.method == m)].groupby("scenario").coverage95.min().loc[scen]
        lo = cp[(cp.method == m)].groupby("scenario").cov95_cp_low.min().loc[scen]
        a.plot(range(len(scen)), c, "o-", color=f"C{k}", ms=3, label=_short(m))
        a.plot(range(len(scen)), lo, "_", color=f"C{k}", ms=8)
    ax[1, 0].axhline(0.95, color="k", lw=0.8)
    ax[1, 0].axhline(0.94, color="k", lw=0.6, ls="--")
    ax[1, 0].set_xticks(range(len(scen)), scen)
    ax[1, 0].set_title("各情景最差比较的 95% 区间覆盖率（短横=CP 下界）")
    ax[1, 0].legend(fontsize=7)
    for k, m in enumerate(meths):
        f = fw[(fw.method == m) & (fw.alpha_family == 0.05)].groupby("scenario").simult_coverage.min().loc[scen]
        ax[1, 1].plot(range(len(scen)), f, "o-", color=f"C{k}", ms=3, label=_short(m))
    ax[1, 1].axhline(0.95, color="k", lw=0.8)
    ax[1, 1].set_xticks(range(len(scen)), scen)
    ax[1, 1].set_title("同时覆盖率（Bonferroni 水平区间，最差零假设类型）")
    vv = pd.read_csv(CAL / "validation_verdicts_by_method.csv")
    a = ax[1, 2]
    a.axis("off")
    lines = [f"预定规则选出的主要方法：{_short(chosen) if chosen else '无'}", ""]
    for m in meths[:2]:
        s = vv[vv.method == m]
        lines.append(f"{_short(m)}{'' if chosen else '（仅展示）'}")
        lines += [f"  {r.scope:<16} α={'' if pd.isna(r.alpha_family) else f'{r.alpha_family:g}'}: {r.verdict}"
                  for r in s.itertuples()]
    a.text(0, 1, "\n".join(lines), va="top", fontsize=6.5)
    fw.to_csv(SRC / "fig5_validation_fwer.csv", index=False)
    fig.tight_layout()
    save(fig, "fig5_validation_fwer_coverage", REP)


def fig6(REP):
    allr = pd.read_csv(REAL / "results_all_methods_all_variants.csv")
    disp, ana = display()
    chosen = ana
    base = allr[(allr.analysis_variant == "main_pooled_nonzero") & (allr.data_version == "原始映射")]
    meths = disp + ["M0_CBB_abs_L5"]
    cond = pd.read_csv(REAL / "conditional_results_display_methods.csv") if (REAL / "conditional_results_display_methods.csv").exists() else None
    scopes = ["pooled_3_sessions", "ses-01_only", "ses-02_only", "ses-03_only", "drop_ses-01", "drop_ses-02", "drop_ses-03"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    a = ax[0]
    for k, m in enumerate(meths):
        d = base[(base.method == m) & (base.session_scope == "pooled_3_sessions")].set_index("direction")
        y = np.arange(6) + (k - 1) * 0.22
        a.errorbar(d.difference_ms, y, xerr=[d.difference_ms - d.ci95_low, d.ci95_high - d.difference_ms], fmt="o",
                   ms=3, color=f"C{k}", label=f"{_short(m)} 95% CI")
        if m == chosen:
            a.errorbar(d.difference_ms, y, xerr=[d.difference_ms - d.ci_bonf_low, d.ci_bonf_high - d.difference_ms],
                       fmt="none", lw=3, alpha=0.3, color=f"C{k}", label=f"{_short(m)} 多重比较水平区间")
            if cond is not None:
                cc = cond[(cond.method == m) & (cond.analysis_variant == "main_pooled_nonzero") &
                          (cond.session_scope == "pooled_3_sessions") & (cond.data_version == "原始映射")].set_index("direction")
                for yy, dname in zip(y, d.index):
                    a.text(max(d.ci_bonf_high.max(), 1.1) + 0.05, yy, cc.loc[dname, "conditional_label"].split("（")[0],
                           fontsize=6, va="center")
    a.set_yticks(range(6), base[base.session_scope == "pooled_3_sessions"].direction.unique())
    a.axvline(0, color="k", lw=0.6)
    a.set_xlabel("池化平均绝对延迟差 Δ（ms）")
    a.set_title("A  主要对象（三会话合并）；右侧文字=条件性标签")
    a.legend(fontsize=6, loc="lower left")
    a = ax[1]
    if chosen:
        for k, sc in enumerate(scopes):
            d = base[(base.method == chosen) & (base.session_scope == sc)].set_index("direction")
            y = np.arange(6) + (k - 3) * 0.11
            a.errorbar(d.difference_ms, y, xerr=[d.difference_ms - d.ci95_low, d.ci95_high - d.difference_ms], fmt="o",
                       ms=2.5, lw=0.7, color=f"C{k}", label=sc)
        a.set_yticks(range(6), d.index)
    a.axvline(0, color="k", lw=0.6)
    a.set_xlabel("Δ（ms），95% CI")
    a.set_title(f"B  分会话与去一会话（{_short(chosen)}；预定规则下无主要方法时仅展示）")
    a.legend(fontsize=6)
    base.to_csv(SRC / "fig6_main_and_session_results.csv", index=False)
    fig.tight_layout()
    save(fig, "fig6_differences_ci_selected", REP)


def fig7(REP):
    allr = pd.read_csv(REAL / "results_all_methods_all_variants.csv")
    _, chosen = display()
    s = allr[allr.method == chosen]
    fig, ax = plt.subplots(1, 3, figsize=(14, 5.5), gridspec_kw=dict(width_ratios=[1, 1, 1.3]))
    var = s[(s.session_scope == "pooled_3_sessions") & (s.data_version == "原始映射")]
    a = ax[0]
    for k, v in enumerate(["main_pooled_nonzero", "A_including_zero", "B_artifact_excluded", "D_pair_equal", "E_trial_median"]):
        d = var[var.analysis_variant == v].set_index("direction")
        y = np.arange(6) + (k - 2) * 0.15
        a.errorbar(d.difference_ms, y, xerr=[d.difference_ms - d.ci95_low, d.ci95_high - d.difference_ms], fmt="o",
                   ms=2.5, lw=0.7, color=f"C{k}", label=v)
    a.set_yticks(range(6), d.index)
    a.axvline(0, color="k", lw=0.6)
    a.set_title(f"A  零值、伪迹、权重与统计量（{_short(chosen)}，95% CI）")
    a.legend(fontsize=6)
    el = s[s.data_version != "原始映射"]
    a = ax[1]
    strata = [e for e in el.electrode_handling.unique() if not e.startswith("去一触点")]
    for k, e in enumerate(strata):
        d = el[el.electrode_handling == e].set_index("direction")
        y = np.arange(6) + (k - len(strata) / 2) * 0.11
        ok = d.pairs_A.fillna(0).to_numpy() > 0
        ok &= d.pairs_B.fillna(0).to_numpy() > 0
        a.errorbar(d.difference_ms[ok], y[ok], xerr=[(d.difference_ms - d.ci95_low)[ok], (d.ci95_high - d.difference_ms)[ok]],
                   fmt="o", ms=2.5, lw=0.7, color=f"C{k}", label=e[:22])
    a.set_yticks(range(6), d.index)
    a.axvline(0, color="k", lw=0.6)
    a.set_title("B  电极处理与分层（未确认敏感性；通道对数见表）")
    a.legend(fontsize=5.5)
    a = ax[2]
    lo = el[el.electrode_handling.str.startswith("去一触点")]
    main = var[var.analysis_variant == "main_pooled_nonzero"].set_index("direction").difference_ms
    for k, (dname, g) in enumerate(lo.groupby("direction", sort=False)):
        a.scatter(g.difference_ms, np.full(len(g), k), s=8, c=np.where(np.sign(g.difference_ms) != np.sign(main[dname]), "r", "0.4"))
        for r in g.itertuples():
            if np.sign(r.difference_ms) != np.sign(main[dname]):
                a.annotate(f"{r.electrode_handling.replace('去一触点：', '')}（−{int(r.pairs_removed_A)}/{int(r.pairs_removed_B)} 对）",
                           (r.difference_ms, k), xytext=(2, 4), textcoords="offset points", fontsize=5.5, color="r")
        a.scatter(main[dname], k, marker="|", s=200, c="k")
    a.set_yticks(range(6), list(dict.fromkeys(lo.direction)))
    a.axvline(0, color="k", lw=0.6)
    a.set_title("C  去一触点（红=变号；括号=两脑区各删除的通道对数）")
    s.to_csv(SRC / "fig7_selected_method_all_rows.csv", index=False)
    fig.tight_layout()
    save(fig, "fig7_sensitivity_electrode_session_weight", REP)


def fig8(REP):
    ovn = pd.read_csv(REAL / "old_vs_new_significance.csv")
    cols = ["comparison", "difference_ms", "old_window_welch_bonferroni_p", "prev_round_M0_L5_B50000_adj_p",
            "M0_CBB_abs_L5_adj_p", "M0_L5_validation_FWER_C1_G_a0.05"] + \
        [c for c in ovn.columns if c.endswith("_adj_p") and c.startswith(tuple(display()[0]))] + \
        [c for c in ovn.columns if c.endswith("_conditional")] + ["selected_decision"]
    cols = [c for c in cols if c in ovn.columns]
    t = ovn[cols].copy()
    for c in t.columns:
        if t[c].dtype.kind == "f":
            t[c] = t[c].map(lambda v: "NA" if pd.isna(v) else (f"{v:.2e}" if abs(v) < 1e-3 and v != 0 else f"{v:.4g}"))
    hdr = {**{c: _short(c.replace("_adj_p", "")) + "\np_adj" for c in t.columns if c.endswith("_adj_p") and c.startswith(("M1", "M2", "M3"))},
           **{c: _short(c.replace("_conditional", "")) + "\n条件性标签" for c in t.columns if c.endswith("_conditional")},"comparison": "比较", "difference_ms": "Δ (ms)", "old_window_welch_bonferroni_p": "旧：窗口 Welch\nBonferroni p",
           "prev_round_M0_L5_B50000_adj_p": "上一轮 M0 L=5\np_adj", "M0_CBB_abs_L5_adj_p": "本轮 M0 L=5\np_adj",
           "M0_L5_validation_FWER_C1_G_a0.05": "M0 L=5 验证\nFWER(C1)", "selected_adj_p": "新主要方法\np_adj",
           "selected_ci95": "新 95% CI", "selected_decision": "预定规则判定"}
    fig, ax = plt.subplots(figsize=(17, 2.8))
    ax.axis("off")
    tb = ax.table(cellText=t.to_numpy(), colLabels=[hdr.get(c, c) for c in t.columns], loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(6.5)
    tb.scale(1, 1.8)
    ax.set_title(f"新旧显著性对照（预定规则选出的主要方法：{load_state().get('selected_method') or '无'}）")
    t.to_csv(SRC / "fig8_old_vs_new_table.csv", index=False)
    save(fig, "fig8_old_vs_new", REP)


def make_all(REP=None):
    for f in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8):
        try:
            f(REP)
        except Exception as e:  # noqa: BLE001
            if REP:
                REP.line(f"  图 {f.__name__} 失败：{type(e).__name__}: {e}")
            raise
