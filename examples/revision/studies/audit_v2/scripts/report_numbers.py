"""Builds the old-vs-new significance table and an auto-generated numeric summary (Chinese)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import infer as I
from common import CAL, ELE, OUT, PREV, REAL, REP_DIR, TAB, WQC, load_state


def md_table(df, floatfmt=None, index=False):
    """Markdown table without the tabulate package."""
    if df is None or len(df) == 0:
        return "*(空表)*"
    out = df.copy()
    if index:
        out = out.reset_index()
    cols = []
    for c in out.columns:
        if isinstance(c, tuple):
            cols.append("-".join("" if x is None else str(x) for x in c).strip("-"))
        else:
            cols.append(str(c))
    out.columns = cols

    def cell(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "NA"
        if isinstance(v, (float, np.floating)):
            if floatfmt:
                try:
                    return format(float(v), floatfmt.replace(":", "").replace("%", "") if False else floatfmt)
                except ValueError:
                    return f"{float(v):.4g}"
            return f"{float(v):.4g}"
        if isinstance(v, (np.integer,)):
            return str(int(v))
        return str(v)

    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(cell(out.iloc[i][c]) for c in cols) + " |" for i in range(len(out))]
    return "\n".join([header, sep] + rows)


def _p(p):
    if p is None or not np.isfinite(p):
        return "NA"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


COND_SCEN = ["A", "B1", "C1", "C2"]


def display_methods():
    """Display methods chosen from the calibration set only (deviations item 8): per group, min of max core FWER."""
    fw = pd.read_csv(CAL / "calibration_fwer.csv")
    f = fw[(fw.alpha_family == 0.05) & fw.scenario.isin(["A", "B1", "B2", "C1", "C2"])]
    mx = f.groupby("method").fwer.max()
    boot = mx[[m for m in mx.index if m.startswith(("M0", "M2"))]].idxmin()
    ana = mx[[m for m in mx.index if m.startswith(("M1", "M3"))]].idxmin()
    out = dict(bootstrap=boot, analytic=ana, max_core_fwer={boot: float(mx[boot]), ana: float(mx[ana])})
    (CAL / "display_methods.json").write_text(json.dumps(out, indent=2))
    return [boot, ana]


def conditional():
    import re
    disp = display_methods()
    vf = pd.read_csv(CAL / "validation_fwer.csv")
    cond = {}
    for m in disp:
        for al in I.ALPHAS:
            f = vf[(vf.method == m) & (vf.alpha_family == al) & vf.scenario.isin(COND_SCEN)]
            cond[(m, al)] = bool(len(f) == 12 and (f.verdict == "通过").all())
    allr = pd.read_csv(REAL / "results_all_methods_all_variants.csv")
    r = allr[allr.method.isin(disp)].copy()
    r["lag1"] = r.diagnostics.map(lambda s: float(re.search(r"ACF\(影响值\)=([+-]?\d+\.\d+)", s).group(1)))
    lab = []
    for x in r.itertuples():
        if not x.test_performed or not np.isfinite(x.adjusted_p):
            lab.append("不检验/无数据")
            continue
        lv = [al for al in (0.001, 0.01, 0.05) if cond[(x.method, al)] and x.adjusted_p < al]
        if x.lag1 > 0.3:
            lab.append(f"条件性证据不足（影响值 lag-1={x.lag1:+.2f} > 0.3）")
        elif lv:
            lab.append(f"条件性显著（p_adj<{lv[0]:g}）")
        elif x.adjusted_p < 0.05:
            lab.append("条件性证据不足（该水平未条件性通过）")
        elif cond[(x.method, 0.05)]:
            lab.append("条件性不显著（不等于无差异）")
        else:
            lab.append("条件性证据不足")
    r["conditional_label"] = lab
    r["conditional_validation"] = r.method.map(lambda m: "; ".join(
        f"α={al:g}:{'通过' if cond[(m, al)] else '未通过'}" for al in I.ALPHAS))
    r.to_csv(REAL / "conditional_results_display_methods.csv", index=False)
    pd.DataFrame([dict(method=m, alpha_family=al, conditional_pass_A_B1_C1_C2=v) for (m, al), v in cond.items()]).to_csv(
        CAL / "conditional_validation_display_methods.csv", index=False)
    return r, disp, cond


def old_vs_new():
    st = load_state()
    chosen = st.get("selected_method")
    welch = pd.read_csv(PREV / "tables/window_level_welch_audit_only.csv")
    welch = welch[welch.scope == "pooled_3_sessions"]
    prev = pd.read_csv(PREV / "tables/primary_result_pooled_L5.csv")
    allr = pd.read_csv(REAL / "results_all_methods_all_variants.csv")
    main = allr[(allr.analysis_variant == "main_pooled_nonzero") & (allr.session_scope == "pooled_3_sessions") &
                (allr.data_version == "原始映射")]
    vf = pd.read_csv(CAL / "validation_fwer.csv")
    rows = []
    for k, (a, b) in enumerate(I.COMPARISONS):
        ra, rb = I.REGIONS[a], I.REGIONS[b]
        w = welch[(welch.region_A == ra) & (welch.region_B == rb)].iloc[0]
        pv = prev[(prev.region_A == ra) & (prev.region_B == rb)].iloc[0]
        m = main[(main.region_A == ra) & (main.region_B == rb)].set_index("method")
        d = dict(comparison=f"{I.SHORT[ra]} − {I.SHORT[rb]}", difference_ms=float(m.difference_ms.iloc[0]),
                 old_window_welch_t=w.welch_t, old_window_welch_bonferroni_p=w.bonferroni_p,
                 old_window_welch_stars=w.stars_old_rule,
                 prev_round_M0_L5_B50000_adj_p=pv.adjusted_p, prev_round_M0_L5_ci95=f"[{pv.ci95_low:+.3f}, {pv.ci95_high:+.3f}]")
        for meth in ["M0_CBB_abs_L1", "M0_CBB_abs_L5", "M2_CBB_studentized_L5", "M2_CBB_studentized_L10",
                     "M1_HAC_fixedb_b0.1", "M1_HAC_fixedb_b0.3", "M1_HAC_fixedb_b1", "M3_HACdetrend_fixedb_b0.3",
                     "M3_HACdetrend_fixedb_b1"]:
            if meth in m.index:
                d[f"{meth}_adj_p"] = float(m.loc[meth, "adjusted_p"])
        for meth in ["M0_CBB_abs_L5"]:
            f = vf[(vf.method == meth) & (vf.alpha_family == 0.05) & (vf.scenario == "C1") & (vf.null_type == "G")]
            if len(f):
                d["M0_L5_validation_FWER_C1_G_a0.05"] = float(f.fwer.iloc[0])
        if chosen and chosen in m.index:
            r = m.loc[chosen]
            d.update(selected_method=chosen, selected_ci95=f"[{r.ci95_low:+.3f}, {r.ci95_high:+.3f}]",
                     selected_ci_bonf=f"[{r.ci_bonf_low:+.3f}, {r.ci_bonf_high:+.3f}]",
                     selected_adj_p=float(r.adjusted_p), selected_decision=r.decision,
                     selected_limitations=r.limitations if isinstance(r.limitations, str) else "")
        else:
            d.update(selected_method="无", selected_decision="证据不足（没有方法通过校准筛选）")
        rows.append(d)
    cr, disp, _ = conditional()
    cm = cr[(cr.analysis_variant == "main_pooled_nonzero") & (cr.session_scope == "pooled_3_sessions") &
            (cr.data_version == "原始映射")]
    for d in rows:
        for m in disp:
            x = cm[(cm.method == m) & (cm.direction == d["comparison"])].iloc[0]
            d[f"{m}_ci95"] = f"[{x.ci95_low:+.3f}, {x.ci95_high:+.3f}]"
            d[f"{m}_conditional"] = x.conditional_label
    df = pd.DataFrame(rows)
    df.to_csv(REAL / "old_vs_new_significance.csv", index=False)
    return df


def write(REP=None):
    ovn = old_vs_new()
    st = load_state()
    chosen = st.get("selected_method")
    L = []
    L.append("# 自动生成的数值汇总（本轮校准与电极核验）\n")
    L.append(f"主要推断方法（按预定规则选出）：**{chosen or '无'}**\n")
    sel = pd.read_csv(CAL / "method_selection.csv")
    L.append("## 方法筛选（校准集）\n")
    L.append(md_table(sel, floatfmt=".4f"))
    vv = pd.read_csv(CAL / "validation_verdicts_by_method.csv")
    L.append("\n## 独立验证判定（全部方法）\n")
    L.append(md_table(vv, floatfmt=".4f"))
    vf = pd.read_csv(CAL / "validation_fwer.csv")
    L.append("\n## 独立验证 FWER（α=0.05）\n")
    piv = vf[vf.alpha_family == 0.05].pivot_table(index="method", columns=["scenario", "null_type"], values="fwer")
    L.append(md_table(piv, floatfmt=".3f", index=True))
    L.append("\n## 新旧显著性对照\n")
    L.append(md_table(ovn, floatfmt=".4g"))
    allr = pd.read_csv(REAL / "results_all_methods_all_variants.csv")
    cr = pd.read_csv(REAL / "conditional_results_display_methods.csv")
    L.append("\n## 条件性分析（偏离说明第 8 条；不替代预定规则）\n")
    L.append(md_table(pd.read_csv(CAL / "conditional_validation_display_methods.csv")))
    cols = ["method", "data_version", "electrode_handling", "analysis_variant", "session_scope", "direction",
            "difference_ms", "ci95_low", "ci95_high", "adjusted_p", "lag1", "conditional_label"]
    L.append(md_table(cr[cols], floatfmt=".4g"))
    if chosen:
        s = allr[(allr.method == chosen)]
        L.append("\n## 主要方法：全部变体、范围与电极处理\n")
        cols = ["data_version", "electrode_handling", "analysis_variant", "session_scope", "direction", "difference_ms",
                "ci95_low", "ci95_high", "adjusted_p", "decision", "pairs_A", "pairs_B", "limitations"]
        cols = [c for c in cols if c in s.columns]
        L.append(md_table(s[cols], floatfmt=".4g"))
    L.append("\n## 电极核验结论\n")
    L.append(md_table(pd.read_csv(ELE / "electrode_findings.csv")))
    L.append("\n## 波形检查\n")
    L.append(md_table(pd.read_csv(WQC / "pair_metrics_summary.csv"), floatfmt=".3f"))
    L.append("\n## 程序核验\n")
    L.append(md_table(pd.read_csv(TAB / "program_validation.csv")))
    (REP_DIR / "results_numeric_summary_CN.md").write_text("\n".join(L), encoding="utf-8")
    if REP:
        REP.line("已写出 reports/results_numeric_summary_CN.md 与 realdata/old_vs_new_significance.csv")
