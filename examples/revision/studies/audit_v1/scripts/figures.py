"""Figures, re-plot source data and the numeric summary for the review 2.4.2 re-test."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import stats_core as sc  # noqa: E402

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
SHORT = sc.REGION_SHORT
VAR_LABEL = {
    "main_pooled_nonzero_all_trials": "Main: pooled, nonzero, all trials",
    "A_pooled_including_zero": "A: pooled, zeros included",
    "B_pooled_artifact_excluded": "B: pooled, artifact trials excluded",
    "D_pair_trial_session_equal_weight": "D: pair/trial/session equal weight",
    "E_trial_median_difference": "E: trial-median difference",
    "F_trend_condition_adjusted": "F: trend + task-condition adjusted",
}
SCOPE_LABEL = {"pooled_3_sessions": "Pooled (3 sessions)", "ses-01_only": "Session 1", "ses-02_only": "Session 2",
               "ses-03_only": "Session 3", "drop_ses-01": "Drop session 1", "drop_ses-02": "Drop session 2",
               "drop_ses-03": "Drop session 3"}


def cname(a, b):
    return f"{SHORT[a]}−{SHORT[b]}"


def save(fig, FIG: Path, name: str):
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def fmt_p(p, B):
    if not np.isfinite(p):
        return "n/a"
    floor = 1 / (B + 1)
    if p <= floor * 1.0000001:
        return f"≤{floor:.1e}"
    return f"{p:.2g}" if p < 0.01 else f"{p:.3f}"


def make_all(OUT: Path, REP):
    TAB, FIG = OUT / "tables", OUT / "figures"
    SRC = FIG / "source_data"
    res = pd.read_csv(TAB / "bootstrap_results_all_settings.csv")
    main = res[res.run_tag == "main"]
    prim = main[main.is_primary].copy()
    desc = pd.read_csv(TAB / "window_level_descriptives_by_scope.csv")
    welch = pd.read_csv(TAB / "window_level_welch_audit_only.csv")
    diag = pd.read_csv(TAB / "diagnostics_block_bootstrap_applicability.csv")
    acf = pd.read_csv(TAB / "diagnostics_acf_trial_level_differences.csv")
    series = pd.read_csv(SRC / "fig5_trial_series.csv")
    refine = res[res.run_tag == "refine"]

    # ---------------------------------------------------------------- Fig 1
    REP.set_task("Fig 1 原图 4c 风格分布图")
    sub = np.load(SRC / "fig1_violin_window_subsample_200k_per_region_seed0.npz")
    d0 = desc[desc.scope == "pooled_3_sessions"].set_index("region")
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    data = [sub[SHORT[r]] for r in sc.REGIONS]
    parts = ax.violinplot(data, showextrema=False, widths=0.8)
    for pc, col in zip(parts["bodies"], plt.cm.Blues(np.linspace(0.35, 0.8, 4))):
        pc.set_facecolor(col); pc.set_edgecolor("0.3"); pc.set_alpha(0.9)
    for i, r in enumerate(sc.REGIONS, start=1):
        q = d0.loc[r]
        ax.vlines(i, q.q25, q.q75, color="0.15", lw=5)
        ax.scatter([i], [q.median_ms], color="white", edgecolor="black", s=36, zorder=4, label="Median (all windows)" if i == 1 else None)
        ax.scatter([i], [q.mean_ms], marker="D", color="crimson", s=30, zorder=5, label="Pooled mean (tested)" if i == 1 else None)
    ymax = 15.625
    pr = prim.set_index(["region_A", "region_B"])
    order = sorted(sc.COMPARISONS, key=lambda ab: (ab[1] - ab[0], ab[0]))
    for n, (a, b) in enumerate(order):
        r = pr.loc[(sc.REGIONS[a], sc.REGIONS[b])]
        y = ymax + 0.6 + n * 0.75
        ax.plot([a + 1, a + 1, b + 1, b + 1], [y, y + 0.2, y + 0.2, y], color="black", lw=1.2)
        ax.text((a + b + 2) / 2, y + 0.22, f"{r.significance_bonferroni}  (Δ={r.difference_ms:+.2f} ms)",
                ha="center", va="bottom", fontsize=8.5)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels([f"{r}\n(n windows={int(d0.loc[r].n_nonzero):,})" for r in sc.REGIONS], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Absolute time-varying delay (ms)")
    ax.set_ylim(0, ymax + 0.8 + 6 * 0.75)
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    ax.set_title("Sub-01, sessions 1–3, 150 trials: window-level |delay| (nonzero)", fontsize=11)
    cap = ("Violins: window-level distribution shown for display only (density from 200,000 random windows per region, seed 0; "
           "box = IQR, white dot = median of all windows).\nStars: Bonferroni-adjusted (6 comparisons) p from the pre-specified "
           "session-stratified, region-synchronous circular block bootstrap of whole trials (L = 5 trials, B = 50,000),\n"
           "testing the difference in POOLED MEAN absolute delay (red diamonds), not medians. *** p<0.001, ** p<0.01, * p<0.05, "
           "ns not significant. Old window-level Welch stars are not used.\nOne participant; trials are not participants; "
           "inference is limited to these electrodes and sessions.")
    fig.text(0.0, -0.12, cap, fontsize=7.2, ha="left", va="top")
    save(fig, FIG, "fig1_fig4c_style_distribution_new_test")
    d0.reset_index().to_csv(SRC / "fig1_region_summary_all_windows.csv", index=False)
    prim.to_csv(SRC / "fig1_fig2_primary_test_results.csv", index=False)

    # ---------------------------------------------------------------- Fig 2
    REP.set_task("Fig 2 六组差异及置信区间")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    labs = []
    for k, r in enumerate(prim.itertuples()):
        y = 5 - k
        ax.plot([r.ci_bonf_low, r.ci_bonf_high], [y, y], color="0.6", lw=6, solid_capstyle="butt",
                label="Bonferroni-level interval (1−0.05/6)" if k == 0 else None)
        ax.plot([r.ci95_low, r.ci95_high], [y, y], color="navy", lw=2.2, label="95% interval (unadjusted)" if k == 0 else None)
        ax.scatter([r.difference_ms], [y], color="crimson", zorder=4, s=30)
        labs.append(f"{cname(r.region_A, r.region_B)}\np_adj {fmt_p(r.adjusted_p, r.bootstrap_B)} {r.significance_bonferroni}")
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(5, -1, -1)); ax.set_yticklabels(labs, fontsize=8.5)
    ax.set_xlabel("Difference in pooled mean |delay| (ms), region A − region B")
    ax.set_title("Primary test: block bootstrap of whole trials (L=5, B=50,000)", fontsize=10.5)
    ax.legend(fontsize=8, loc="best")
    save(fig, FIG, "fig2_primary_differences_ci")

    # ---------------------------------------------------------------- Fig 3
    REP.set_task("Fig 3 各会话及去一会话差异")
    sel = main[(main.analysis_variant == "main_pooled_nonzero_all_trials") & (main.block_length == 5)]
    scopes = ["pooled_3_sessions", "ses-01_only", "ses-02_only", "ses-03_only", "drop_ses-01", "drop_ses-02", "drop_ses-03"]
    cols = ["black", "tab:blue", "tab:orange", "tab:green", "0.45", "0.6", "0.75"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for k, (a, b) in enumerate(sc.COMPARISONS):
        ax = axes.ravel()[k]
        s_ = sel[(sel.region_A == sc.REGIONS[a]) & (sel.region_B == sc.REGIONS[b])].set_index("session_scope")
        for j, (scp, c) in enumerate(zip(scopes, cols)):
            r = s_.loc[scp]
            ax.plot([j, j], [r.ci_bonf_low, r.ci_bonf_high], color=c, lw=5, alpha=0.35)
            ax.plot([j, j], [r.ci95_low, r.ci95_high], color=c, lw=2)
            ax.scatter([j], [r.difference_ms], color=c, zorder=3)
            ax.text(j, r.ci_bonf_high, r.significance_bonferroni, ha="center", va="bottom", fontsize=7)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xticks(range(len(scopes))); ax.set_xticklabels([SCOPE_LABEL[s] for s in scopes], rotation=45, ha="right", fontsize=7.5)
        ax.set_title(cname(sc.REGIONS[a], sc.REGIONS[b]), fontsize=10)
        if k % 3 == 0:
            ax.set_ylabel("Δ pooled mean |delay| (ms)")
    fig.suptitle("Session-specific and leave-one-session-out differences (main estimand, L=5; thin = 95%, thick = Bonferroni-level; "
                 "stars = Bonferroni within each scope)", fontsize=9.5)
    fig.tight_layout()
    save(fig, FIG, "fig3_session_specific_and_leave_one_out")
    sel.to_csv(SRC / "fig3_session_and_loso_results.csv", index=False)

    # ---------------------------------------------------------------- Fig 4 + sensitivity table
    REP.set_task("Fig 4 敏感性分析")
    pooled = main[main.session_scope == "pooled_3_sessions"]
    variants = list(VAR_LABEL)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    for k, (a, b) in enumerate(sc.COMPARISONS):
        ax = axes.ravel()[k]
        s_ = pooled[(pooled.region_A == sc.REGIONS[a]) & (pooled.region_B == sc.REGIONS[b])]
        ylabels = []
        y = 0
        for v in variants:
            for L in (1, 2, 5, 10):
                r = s_[(s_.analysis_variant == v) & (s_.block_length == L)].iloc[0]
                c = "crimson" if r.adjusted_p < 0.05 else "0.4"
                ax.plot([r.ci_bonf_low, r.ci_bonf_high], [-y, -y], color=c, lw=4, alpha=0.3)
                ax.plot([r.ci95_low, r.ci95_high], [-y, -y], color=c, lw=1.6)
                ax.scatter([r.difference_ms], [-y], color=c, s=14 if L != 5 else 30, marker="o" if L != 5 else "D", zorder=3)
                ylabels.append(f"{v.split('_')[0]} L={L}")
                y += 1
            y += 0.6
        ax.axvline(0, color="black", lw=0.7)
        ax.set_title(cname(sc.REGIONS[a], sc.REGIONS[b]), fontsize=10)
        ticks = []
        yy = 0
        for v in variants:
            for L in (1, 2, 5, 10):
                ticks.append(-yy); yy += 1
            yy += 0.6
        ax.set_yticks(ticks); ax.set_yticklabels(ylabels, fontsize=6.5)
        ax.set_xlabel("Difference (ms)", fontsize=8)
    fig.suptitle("Sensitivity (pooled 3 sessions): block length × zero handling × artifact handling × weighting × median × adjustment.\n"
                 "Red = Bonferroni p_adj<0.05 within that setting; diamond = pre-specified L=5; thin 95%, thick Bonferroni-level interval. "
                 "Estimands differ across variants (see table).", fontsize=9)
    fig.tight_layout()
    save(fig, FIG, "fig4_sensitivity_block_length_zero_artifact_weights")
    wide = []
    for (v, scp, a, b), g in main.groupby(["analysis_variant", "session_scope", "region_A", "region_B"], sort=False):
        row = dict(analysis_variant=v, session_scope=scp, comparison=cname(a, b),
                   difference_ms=g.difference_ms.iloc[0], estimand=g.estimand.iloc[0])
        for L in (1, 2, 5, 10):
            r = g[g.block_length == L].iloc[0]
            row[f"ci95_L{L}"] = f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]"
            row[f"p_adj_L{L}"] = r.adjusted_p
            row[f"sig_L{L}"] = r.significance_bonferroni
        wide.append(row)
    wide = pd.DataFrame(wide)
    wide.to_csv(TAB / "sensitivity_summary_wide.csv", index=False)
    pooled.to_csv(SRC / "fig4_sensitivity_pooled.csv", index=False)

    # ---------------------------------------------------------------- Fig 5 diagnostics
    REP.set_task("Fig 5 试次序列与自相关诊断")
    fig = plt.figure(figsize=(15, 13))
    gs = fig.add_gridspec(4, 6, height_ratios=[1, 1, 1, 0.9], hspace=0.55, wspace=0.35)
    for j, s in enumerate(["ses-01", "ses-02", "ses-03"]):
        ax = fig.add_subplot(gs[0, 2 * j:2 * j + 2])
        g = series[series.session == s]
        for r in sc.REGIONS:
            ax.plot(g.trial_pos, g[f"mean_{SHORT[r]}"], marker=".", lw=1, label=SHORT[r])
        art = g[g.artifact]
        ax.scatter(art.trial_pos, np.full(len(art), ax.get_ylim()[0]), marker="^", color="red", s=18, label="artifact trial")
        ax.set_title(f"{s}: trial-level pooled mean |delay|", fontsize=9)
        ax.set_xlabel("trial (time order)", fontsize=8)
        if j == 0:
            ax.set_ylabel("ms"); ax.legend(fontsize=6.5, ncol=3)
    dcols = [c for c in series.columns if c.startswith("diff_")]
    for k, c in enumerate(dcols):
        ax = fig.add_subplot(gs[1 + k // 3, 2 * (k % 3):2 * (k % 3) + 2])
        x = np.arange(len(series))
        ax.plot(x, series[c], color="0.3", lw=0.9, marker=".", ms=3)
        for s, col in zip(["ses-01", "ses-02", "ses-03"], ["tab:blue", "tab:orange", "tab:green"]):
            m = (series.session == s).to_numpy()
            ax.plot(x[m], np.poly1d(np.polyfit(x[m], series[c][m], 1))(x[m]), color=col, lw=1.6)
            ax.hlines(series[c][m].mean(), x[m][0], x[m][-1], color=col, lw=0.8, ls="--")
        for bnd in (49.5, 99.5):
            ax.axvline(bnd, color="black", lw=0.6)
        ax.axhline(0, color="black", lw=0.5, ls=":")
        art = series.artifact.to_numpy()
        ax.scatter(x[art], series[c][art], color="red", s=10, zorder=3)
        ax.set_title(f"{c[5:]}: trial-level Δ (red = artifact; solid = trend, dashed = mean)", fontsize=7.5)
        ax.set_xlabel("trial index (sessions concatenated; blocks never cross sessions)", fontsize=7)
    for k, c in enumerate(dcols):
        ax = fig.add_subplot(gs[3, k])
        g = acf[acf.comparison == c[5:]]
        ax.bar(g.lag, g.acf, color=np.where(g.exceeds, "crimson", "0.5"))
        ax.axhline(g.band_95.iloc[0], color="black", ls="--", lw=0.7); ax.axhline(-g.band_95.iloc[0], color="black", ls="--", lw=0.7)
        for L in (2, 5, 10):
            ax.axvline(L + 0.5, color="tab:blue", lw=0.6, ls=":")
        ax.set_title(f"ACF {c[5:]}", fontsize=8); ax.set_ylim(-0.6, 0.8); ax.set_xlabel("lag (trials)", fontsize=7)
    fig.suptitle("Diagnostics for trial-level block bootstrap. ACF pooled within sessions (session means removed); dashed = ±1.96/√150; "
                 "dotted = block lengths 2, 5, 10.\nA non-significant ACF is not evidence of independence.", fontsize=9.5)
    save(fig, FIG, "fig5_trial_series_and_acf")
    acf.to_csv(SRC / "fig5_acf.csv", index=False)

    fig, axes = plt.subplots(3, 6, figsize=(16, 8))
    for k, c in enumerate(dcols):
        for i, col in enumerate(["set_size", "correct", "artifact"]):
            ax = axes[i, k]
            lv = sorted(series[col].unique())
            vals = [series[c][series[col] == l].dropna().to_numpy() for l in lv]
            ax.boxplot(vals, tick_labels=[str(l) for l in lv], widths=0.6, showfliers=False)
            for j, v in enumerate(vals, start=1):
                ax.scatter(np.full(v.size, j) + np.random.default_rng(0).uniform(-0.15, 0.15, v.size), v, s=5, alpha=0.5)
            ax.axhline(0, color="black", lw=0.5, ls=":")
            if i == 0:
                ax.set_title(c[5:], fontsize=9)
            if k == 0:
                ax.set_ylabel(col, fontsize=9)
    fig.suptitle("Trial-level region differences (ms) by task condition, behaviour and artifact flag (descriptive)", fontsize=10)
    fig.tight_layout()
    save(fig, FIG, "fig5b_task_covariates")

    # ---------------------------------------------------------------- Fig 6 Welch vs new
    REP.set_task("Fig 6 原 Welch 与新检验对照表")
    wp = welch[welch.scope == "pooled_3_sessions"].set_index(["region_A", "region_B"])
    rows = []
    for a, b in sc.COMPARISONS:
        A, Bn = sc.REGIONS[a], sc.REGIONS[b]
        w = wp.loc[(A, Bn)]
        row = dict(comparison=cname(A, Bn), window_mean_difference_ms=w.difference_ms, welch_t=w.welch_t,
                   welch_df_basis=f"{int(w.n_A):,} vs {int(w.n_B):,} windows", welch_bonferroni_p=w.bonferroni_p,
                   welch_stars=w.stars_old_rule)
        for L in (1, 2, 5, 10):
            r = pooled[(pooled.analysis_variant == "main_pooled_nonzero_all_trials") & (pooled.block_length == L) &
                       (pooled.region_A == A) & (pooled.region_B == Bn)].iloc[0]
            if L == 5:
                row["bootstrap_difference_ms"] = r.difference_ms
                row["bootstrap_ci95_L5"] = f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]"
                row["bootstrap_ci_bonf_L5"] = f"[{r.ci_bonf_low:+.3f}, {r.ci_bonf_high:+.3f}]"
            row[f"bootstrap_p_adj_L{L}"] = r.adjusted_p
            row[f"bootstrap_sig_L{L}"] = r.significance_bonferroni
        row["unit_of_resampling"] = "whole trials (150; 3 sessions × 50), blocks within session"
        rows.append(row)
    cmp_ = pd.DataFrame(rows)
    cmp_.to_csv(TAB / "welch_vs_new_test_comparison.csv", index=False)
    fig, ax = plt.subplots(figsize=(15, 3.2))
    ax.axis("off")
    show = cmp_[["comparison", "window_mean_difference_ms", "welch_t", "welch_bonferroni_p", "welch_stars",
                 "bootstrap_ci95_L5", "bootstrap_p_adj_L1", "bootstrap_p_adj_L2", "bootstrap_p_adj_L5",
                 "bootstrap_p_adj_L10", "bootstrap_sig_L5"]].copy()
    B = int(pooled.bootstrap_B.iloc[0])
    show["window_mean_difference_ms"] = show.window_mean_difference_ms.map(lambda v: f"{v:+.3f}")
    show["welch_t"] = show.welch_t.map(lambda v: f"{v:.1f}")
    show["welch_bonferroni_p"] = show.welch_bonferroni_p.map(lambda v: f"{v:.1e}")
    for L in (1, 2, 5, 10):
        show[f"bootstrap_p_adj_L{L}"] = show[f"bootstrap_p_adj_L{L}"].map(lambda v: fmt_p(v, B) if v < 1 else "1")
    show.columns = ["Comparison", "Δ mean (ms)", "Welch t (windows)", "Welch Bonf. p", "Old stars", "Bootstrap 95% CI (L=5)",
                    "p_adj L=1", "p_adj L=2", "p_adj L=5 (ref)", "p_adj L=10", "New stars (L=5)"]
    tb = ax.table(cellText=show.values, colLabels=show.columns, loc="center", cellLoc="center")
    tb.auto_set_font_size(False); tb.set_fontsize(8); tb.scale(1, 1.5)
    ax.set_title("Window-level Welch (audit only; dependent windows treated as independent) vs trial-level block bootstrap", fontsize=10)
    save(fig, FIG, "fig6_welch_vs_new_table")
    show.to_csv(SRC / "fig6_table_display.csv", index=False)

    # ---------------------------------------------------------------- numeric summary
    REP.set_task("数字摘要")
    lines = ["# 数字摘要（自动生成，供报告引用）", "",
             "本文件由 `scripts/figures.py` 从结果表直接生成，不含解释。", "",
             "## 主分析（池化、非零、全部 150 试次、L=5、B=50000）", "",
             "| 比较 | Δ (ms) | 95% 区间 | Bonferroni 水平区间 | p_raw | p_adj | MC SE(p_raw) | 显著性 |", "|---|---|---|---|---|---|---|---|"]
    for r in prim.itertuples():
        lines.append(f"| {cname(r.region_A, r.region_B)} | {r.difference_ms:+.4f} | [{r.ci95_low:+.4f}, {r.ci95_high:+.4f}] | "
                     f"[{r.ci_bonf_low:+.4f}, {r.ci_bonf_high:+.4f}] | {fmt_p(r.raw_p, r.bootstrap_B)} | "
                     f"{fmt_p(r.adjusted_p, r.bootstrap_B)} | {r.p_mc_se:.1e} | {r.significance_bonferroni} |")
    lines += ["", "## 各设置显著性（Bonferroni p_adj<0.05 的块长）", "",
              "| 版本 | 范围 | 比较 | Δ (ms) | L=1 | L=2 | L=5 | L=10 |", "|---|---|---|---|---|---|---|---|"]
    for r in wide.itertuples():
        lines.append(f"| {r.analysis_variant.split('_')[0]} | {r.session_scope} | {r.comparison} | {r.difference_ms:+.3f} | "
                     + " | ".join(f"{fmt_p(getattr(r, f'p_adj_L{L}'), B)} {getattr(r, f'sig_L{L}')}" for L in (1, 2, 5, 10)) + " |")
    if len(refine):
        lines += ["", "## 加密重算（B=500000）", "", "| 版本 | 范围 | L | 比较 | p_adj (B=50000) | p_adj (B=500000) |", "|---|---|---|---|---|---|"]
        for r in refine.itertuples():
            m = main[(main.analysis_variant == r.analysis_variant) & (main.session_scope == r.session_scope) &
                     (main.block_length == r.block_length) & (main.region_A == r.region_A) & (main.region_B == r.region_B)].iloc[0]
            lines.append(f"| {r.analysis_variant.split('_')[0]} | {r.session_scope} | {r.block_length} | {cname(r.region_A, r.region_B)} | "
                         f"{fmt_p(m.adjusted_p, m.bootstrap_B)} | {fmt_p(r.adjusted_p, r.bootstrap_B)} |")
    lines += ["", "## 诊断", "", "| 比较 | ACF1 | lag6–15 超带数 | SE L10/L5 | SE L5/L1 | 警示 |", "|---|---|---|---|---|---|"]
    for r in diag.itertuples():
        lines.append(f"| {r.comparison} | {r.acf_lag1:+.2f} | {r.n_lags_6_15_beyond_band} | {r.se_ratio_L10_L5:.2f} | "
                     f"{r.se_ratio_L5_L1:.2f} | {r.inference_condition_flags} |")
    lines += ["", "## 窗口级描述（池化）", "", "| 脑区 | 原始窗口 | 零值比例 | 非零 N | 均值 | 中位数 | Q25 | Q75 |", "|---|---|---|---|---|---|---|---|"]
    for r in d0.reset_index().itertuples():
        lines.append(f"| {r.region} | {r.n_windows_raw:,} | {r.zero_fraction:.4f} | {r.n_nonzero:,} | {r.mean_ms:.4f} | "
                     f"{r.median_ms:.5f} | {r.q25:.4f} | {r.q75:.4f} |")
    (OUT / "results_numeric_summary_CN.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    REP.set_task(None)
    REP.line("图表（PNG+PDF）、源数据与数字摘要已写出。")
