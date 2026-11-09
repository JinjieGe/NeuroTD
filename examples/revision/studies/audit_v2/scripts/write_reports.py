"""Write REPORT_CN.md, English Methods, and the reviewer 2.4.2 draft from completed tables."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common import CAL, ELE, LOGS, OUT, REAL, TAB, WQC, load_state
from report_numbers import md_table


def _p(p):
    if p is None or (isinstance(p, float) and not np.isfinite(p)):
        return "NA"
    return f"{p:.2e}" if abs(p) < 1e-3 else f"{p:.4f}"


def _row(df, **kw):
    q = df
    for k, v in kw.items():
        q = q[q[k] == v]
    return q.iloc[0] if len(q) else None


def _fwer(fw, method, scen, ntype="G", alpha=0.05):
    r = _row(fw, method=method, scenario=scen, null_type=ntype, alpha_family=alpha)
    if r is None:
        return "NA"
    return f"{r.fwer:.3f}（CP [{r.fwer_cp_low:.3f}, {r.fwer_cp_high:.3f}]，{r.verdict}）"


def write(REP=None):
    st = load_state()
    chosen = st.get("selected_method")
    fw = pd.read_csv(CAL / "validation_fwer.csv")
    vv = pd.read_csv(CAL / "validation_verdicts_by_method.csv")
    sel = pd.read_csv(CAL / "method_selection.csv")
    cfw = pd.read_csv(CAL / "calibration_fwer.csv")
    ovn = pd.read_csv(REAL / "old_vs_new_significance.csv")
    allr = pd.read_csv(REAL / "results_all_methods_all_variants.csv")
    cr = pd.read_csv(REAL / "conditional_results_display_methods.csv")
    findings = pd.read_csv(ELE / "electrode_findings.csv")
    wqc = pd.read_csv(WQC / "pair_metrics_summary.csv")
    theory = pd.read_csv(TAB / "old_method_theory_cbb_variance_ratio.csv")
    prog = pd.read_csv(TAB / "program_validation.csv")
    time_est = json.loads((LOGS / "time_estimate_at_start.json").read_text())
    timing = pd.read_csv(OUT / "timing_summary.csv") if (OUT / "timing_summary.csv").exists() else pd.DataFrame()
    disp = json.loads((CAL / "display_methods.json").read_text())
    boot_disp, ana_disp = disp["bootstrap"], disp["analytic"]
    condv = pd.read_csv(CAL / "conditional_validation_display_methods.csv")

    main = allr[(allr.analysis_variant == "main_pooled_nonzero") &
                (allr.session_scope == "pooled_3_sessions") &
                (allr.data_version == "原始映射")]
    loo = allr[(allr.analysis_variant == "main_pooled_nonzero") &
               (allr.session_scope == "pooled_3_sessions") &
               (allr.electrode_handling.fillna("").astype(str).str.startswith("去一触点"))]

    def main_row(method, direction):
        q = main[(main.method == method) & (main.direction == direction)]
        return q.iloc[0] if len(q) else None

    directions = ["AMY − HIP", "AMY − MTG", "AMY − STG", "HIP − MTG", "HIP − STG", "MTG − STG"]
    # results table uses "AMY − HIP" or "Amygdala − Hippocampus" — check
    if "direction" in main.columns and len(main):
        directions = list(dict.fromkeys(main.direction.tolist()))

    n_pass = int(sel.passes_screen.sum())
    formal = "证据不足（没有方法通过校准筛选）"
    core_fail_b2 = cfw[(cfw.alpha_family == 0.05) & (cfw.scenario == "B2") & (cfw.null_type == "G")]

    def th(phi, L):
        r = theory[(theory.phi == phi) & (theory.L == L)].iloc[0]
        return r

    t60_l1, t60_l5 = th(0.6, 1), th(0.6, 5)

    # AHL1 leave-one-out
    ahl1 = loo[loo.electrode_handling.str.contains("AHL1", na=False)]
    ahl1_note = ""
    if len(ahl1):
        bits = []
        for r in ahl1[ahl1.method == (chosen or boot_disp)].itertuples():
            bits.append(f"{r.direction} {r.difference_ms:+.3f} ms（删除通道对 A {int(getattr(r, 'pairs_removed_A', -1))}/B {int(getattr(r, 'pairs_removed_B', -1))}）")
        ahl1_note = "；".join(bits[:6])

    # timing
    def stage_sec(name):
        if timing.empty:
            return None
        g = timing[timing.stage == name]
        g = g[g.status.astype(str).str.startswith("完成")]
        return float(g.seconds.sum()) if len(g) else None

    # ---------- Chinese report ----------
    L = []
    L.append("# 图 4c 校准、电极核验与重新检验报告（回应 Reviewer 2.4.2）")
    L.append("")
    L.append(f"运行目录：`{OUT.name}/`")
    L.append(f"分析日期：2026-10-31")
    L.append("性质：在已知上一轮结果之后的补充分析，不是独立前瞻性预注册。方案见 `analysis_plan_calibration_electrodes.md`。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 0. 结论摘要")
    L.append("")
    L.append(f"1. **预定规则下没有方法通过筛选。** 校准集核心情景（A/B1/B2/C1/C2 × G/P1/P2）中，18 个候选方法全部未同时满足“家族 α=0.05 的 FWER CP 上界 ≤ 0.075”和“每个比较 95% 覆盖率 ≥ 0.93”。通过筛选的方法数 = {n_pass}。因此**不指定主要推断方法**。")
    L.append(f"2. **正式判定：六组比较全部为「{formal}」。** 这不是“证明没有差异”。点估计仍是上一轮复现的池化差值（约 0.08–1.00 ms）。")
    L.append("3. **上一轮 L=5、φ=0.6 的约 14% 误报率有解析原因。** 去均值后，circular block bootstrap 的方差期望只有真实长程方差的约 0.57（L=5）或 0.24（L=1）。区间过窄，拒绝率升高。上一轮把 42/300=14.0% 记成 `passed=True`，这一项必须算作不通过，不能算进“12/12 通过”。")
    L.append("4. **学生化与 HAC/fixed-b 在独立、短相关情景中改善明显，但在 φ=0.6 及更强依赖下仍偏宽松或覆盖不足。** 压力情景（更长相关、长记忆、趋势、重尾、异常试次）完整保留，不用来挑选“能显著”的方法。")
    L.append("5. **没有任何比较可以支持 p<0.001 的正式陈述。** bootstrap 方法在家族 0.001 水平因 B=1,999 分辨率不足标为不可评价；即使解析方法在部分情景通过 0.001，也因筛选失败不能用作正式显著性。")
    L.append("6. **“海马较高”依赖 AHL1 这一触点组合。** 去掉 AHL1 会删掉海马 3 对中的 2 对，只剩 AHL2–PHR1。这是组合改变，不是已证实的触点错误。")
    L.append("7. **近乎全零延迟的通道对未被证实为重复通道。** 预定排除规则（逐样本相同 ≥0.99，或恒值/饱和/缺失 ≥50%）在 15 个检查试次上 0 对满足。AL1–AL2 等对相关高、差值仍非零，只能作为未确认敏感性。")
    L.append("8. **电极标签存在明确冲突，不能按脑区一般性质解释图 4c。** 微电极序列引用的是宏触点电极表行；同一束微电极被标成沿电极杆的不同脑区。ECL 的侧别字符串与坐标轴说明冲突。距离不能确认为个体解剖毫米。")
    L.append("9. **对 Reviewer 2.4.2：重叠窗口不当作独立观测这一点已经处理；可靠的显著性还没有。** 建议正文改为描述性差值和区间，删除原 Welch 星号和 p<0.001，并收窄到单受试者、当前触点组合、这三个会话。")
    L.append("")
    L.append("条件性分析（方案偏离第 8 条，**不替代**预定规则）：展示方法 "
             f"bootstrap=`{boot_disp}`、解析=`{ana_disp}`。"
             "仅当独立验证在 A/B1/C1/C2（去掉 B2）全部通过时，才给“条件性显著”标签。正式结论仍是证据不足。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. 上一轮必须正面记录的问题")
    L.append("")
    L.append("来源：`tables/previous_round_errata.csv` 与上一轮 `synthetic_validation/synthetic_validation_results.csv`。")
    L.append("")
    L.append("| 问题 | 处理 |")
    L.append("|---|---|")
    L.append(f"| AR(1) φ=0.6、每会话 50 试次，名义 5% 检验实际拒绝率 L=1 约 34.3%、L=5 约 14.0% | 本轮用独立验证重新估计；解析近似：L=1 单检验拒绝率 {t60_l1.approx_rejection_at_nominal_0_05_single_test:.3f}，L=5 为 {t60_l5.approx_rejection_at_nominal_0_05_single_test:.3f} |")
    L.append("| pipeline 把该项写成 `passed=True`，导致“12/12 通过” | 本轮把该项记为不通过；程序核验通过 ≠ 误报率合格 |")
    L.append("| 去掉 AHL1 后 HIP−MTG +0.695→−0.544 ms，HIP−STG +1.002→−0.237 ms | 本轮去一触点分析保留，并写明删除的通道对数（海马 3 对中删 2 对） |")
    L.append("| 通道对等权后 AMY−STG +0.225→−0.313 ms | 保留为敏感性 D，不与主对象混用 |")
    L.append("| F 变体 L=1 有一次重采样无法拟合，有效 B=49999 | 记入勘误；本轮 M3 去趋势 HAC 不再用“没有秩亏”一笔带过 |")
    L.append("| 部分敏感性 p<0.05 但未到 p<0.001 | 勘误表 46 行；本轮不把 p<0.05 写成 p<0.001 |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 2. 旧方法为何会出现约 14% 的误报率")
    L.append("")
    L.append("上一轮主检验是会话内 circular block bootstrap（中心化绝对误差，L=5，B=50,000）。")
    L.append("对长度为 n=50 的去均值序列，Bartlett 核、块长 L 的 circular block 方差期望小于真实长程方差。")
    L.append(f"AR(1) φ=0.6 时，E[CBB 方差]/真实方差：L=1 为 {t60_l1.expected_cbb_var_over_true_var:.3f}，L=5 为 {t60_l5.expected_cbb_var_over_true_var:.3f}。")
    L.append("标准误被低估后，|Δ̂|/se 偏大，名义 5% 的单检验拒绝率分别约为 "
             f"{t60_l1.approx_rejection_at_nominal_0_05_single_test:.3f} 和 {t60_l5.approx_rejection_at_nominal_0_05_single_test:.3f}，")
    L.append("与上一轮 300 次合成实验的 34.3% 和 14.0% 一致。")
    L.append("")
    L.append("另外：每会话只有 50 个试次；circular 连接把第 50 与第 1 个试次当作相邻；六个比较相关，家族错误率还会更高。")
    L.append("上一轮把“函数能跑完、相同种子可复现、四脑区共用索引”记为通过，那些只是程序检查。")
    L.append("")
    L.append("独立验证中旧方法 M0 L=1 / L=5 在全局零假设下的 FWER（α=0.05）：")
    L.append("")
    L.append("| 情景 | M0 L=1 | M0 L=5 |")
    L.append("|---|---|---|")
    for s in ["A", "B1", "B2", "C1", "C2", "D1", "D2", "D3", "D4", "D5"]:
        L.append(f"| {s} | {_fwer(fw, 'M0_CBB_abs_L1', s)} | {_fwer(fw, 'M0_CBB_abs_L5', s)} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 3. 新方案在哪些范围内改善，哪些情景仍失败")
    L.append("")
    L.append("候选方法：M0 中心化 CBB（L=1/2/5/10）；M2 学生化 CBB（同样的 L）；M1 HAC + fixed-b（b=0.1–1）；M3 去趋势 HAC + fixed-b。")
    L.append("筛选只用校准集；判定用独立验证集。核心情景 = A、B1、B2、C1、C2。D1–D5 是压力测试。")
    L.append("")
    L.append("### 3.1 校准筛选")
    L.append("")
    L.append(md_table(sel[["method", "max_fwer_core", "max_fwer_cp_high_core", "worst_core",
                           "min_coverage_core", "passes_screen", "C1_mean_median_halfwidth"]], floatfmt=".4f"))
    L.append("")
    L.append("### 3.2 独立验证判定（核心 / 压力）")
    L.append("")
    L.append(md_table(vv[vv.scope.isin(["core", "stress", "core_coverage95"])], floatfmt=".4f"))
    L.append("")
    L.append("### 3.3 展示方法在独立验证中的 FWER（α=0.05，全局零假设）")
    L.append("")
    L.append("| 情景 | 旧 M0 L=5 | 展示 bootstrap | 展示解析 |")
    L.append("|---|---|---|---|")
    for s in ["A", "B1", "B2", "C1", "C2", "D1", "D2", "D3", "D4", "D5"]:
        L.append(f"| {s} | {_fwer(fw, 'M0_CBB_abs_L5', s)} | {_fwer(fw, boot_disp, s)} | {_fwer(fw, ana_disp, s)} |")
    L.append("")
    L.append("改善：在独立或短相关（A、B1）以及接近真实残差相关的 C1/C2 上，学生化或较大 b 的 HAC 通常比 M0 L=1/L=5 更接近名义水平。")
    L.append("仍失败：φ=0.6（B2）及更长相关、长记忆近似时，核心筛选要求无法满足；区间覆盖也掉到 0.93 以下。")
    L.append("压力情景的失败全部保留，不删除。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 4. 六组比较现在分别属于显著、不显著还是证据不足")
    L.append("")
    L.append("主对象：μ_r = ΣS/ΣN，非零窗口，150 个试次，三会话合并，原始 60 个区内通道对。")
    L.append("")
    rows = []
    for dname in directions:
        r0 = main_row(boot_disp, dname)
        r1 = main_row(ana_disp, dname)
        c0 = cr[(cr.method == boot_disp) & (cr.direction == dname) &
                (cr.analysis_variant == "main_pooled_nonzero") &
                (cr.session_scope == "pooled_3_sessions") &
                (cr.data_version == "原始映射")]
        o = _row(ovn, comparison=dname) if "comparison" in ovn.columns else None
        if o is None and len(ovn):
            # try matching first column
            o = ovn.iloc[directions.index(dname)] if dname in directions and directions.index(dname) < len(ovn) else None
        rows.append({
            "比较": dname,
            "Δ (ms)": None if r0 is None else f"{r0.difference_ms:+.3f}",
            "展示 bootstrap 95% CI": None if r0 is None else f"[{r0.ci95_low:+.3f}, {r0.ci95_high:+.3f}]",
            "展示解析 95% CI": None if r1 is None else f"[{r1.ci95_low:+.3f}, {r1.ci95_high:+.3f}]",
            "预定规则判定": formal,
            "条件性标签（不替代正式结论）": "" if not len(c0) else c0.iloc[0].conditional_label,
        })
    L.append(md_table(pd.DataFrame(rows)))
    L.append("")
    L.append("**六组在预定规则下全部是证据不足。** 未得到可靠 p 值，不能写成“证明没有差异”。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 5. 哪些比较可以支持 p<0.001，哪些只能支持较弱阈值")
    L.append("")
    L.append("- 预定规则：没有任何比较可以支持 0.05 / 0.01 / 0.001 中的任一正式阈值。")
    L.append("- bootstrap 方法在家族 0.001 水平标为“不可评价”（B=1,999 时最小 p=5×10⁻⁴，Bonferroni 阈值 0.001/6 不可达）。不可评价 ≠ 零误报。")
    L.append("- 条件性分析若在 A/B1/C1/C2 上通过 0.05 或 0.01，也**不能**外推到 φ=0.6 或人群。报告中的条件性标签见 `realdata/conditional_results_display_methods.csv`。")
    L.append("- 上一轮写的 p_adj=1.2×10⁻⁴ 只是 B=50,000 的下限，在误报率未校准的方法上没有 0.001 的含义。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 6. 海马较高的结论是否依赖 AHL1")
    L.append("")
    L.append("是，作为触点组合它依赖 AHL1。海马只有 AHL1、AHL2、PHR1 三个通道、三对。去掉 AHL1 删除 AHL1–AHL2 和 AHL1–PHR1，只剩 AHL2–PHR1。")
    L.append("上一轮池化差值：HIP−MTG +0.695 → −0.544 ms；HIP−STG +1.002 → −0.237 ms。")
    if ahl1_note:
        L.append(f"本轮去一触点（展示方法 {chosen or boot_disp}）：{ahl1_note}。")
    L.append("这不能解释为 AHL1 录错。三对中有两对是跨电极、跨侧标签（AHL 左、PHR 右）。按数据描述，AHL3–AHL8 微电极与 AHL1/AHL2 同在尖端附近，却被标成 STG/MTG。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 7. 电极标签或近零通道问题是否得到证实")
    L.append("")
    L.append("### 7.1 电极核验（分级）")
    L.append("")
    L.append(md_table(findings[["id", "topic", "category"]]))
    L.append("")
    L.append("已确认：波形列通过 ElectricalSeries 引用映射；共同颅内参考；三会话元数据一致。")
    L.append("存在明确冲突：微电极与宏电极共用电极表行；微电极脑区标签沿杆分布；AHL1/AHL2/PHR1 跨半球组合；ECL 与 LR 的侧别字符串与坐标冲突；滤波说明冲突；信号单位（声明 volts，数值像 ±200 满量程 int16）。")
    L.append("证据不足：坐标单位/坐标系；SOZ 中的 AHR；微电极频谱与跨束近零相关的原因。")
    L.append("**没有仅凭命名或距离改左右侧。** 待确认问题见 `electrodes/questions_for_authors.md`。")
    L.append("")
    L.append("### 7.2 近零延迟通道对")
    L.append("")
    L.append(md_table(wqc[["pair", "region", "selection_reason", "identical_fraction_median",
                           "r_zero_lag_median", "diff_rms_ratio_median", "window_zero_fraction_mean",
                           "exclude_in_evidence_corrected"]], floatfmt=".3f"))
    L.append("")
    L.append("0 对满足预定排除规则。高相关 + 高零延迟比例本身不能证明重复通道、串扰或硬件错误。")
    L.append("事后检查：不同试次配对时，AL1–AL2、PHR6–PHR7、ECL5–ECL6 的零延迟比例从 >95% 降到约 0，非零均值升到约 7–9 ms，说明同试次近零依赖同时刻的相似波形，不是通道副本。")
    L.append("证据支持的纠正分析与原始映射相同（排除 0 对）。近零对排除、同侧/异侧、同束/跨束只作为未确认敏感性。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 8. Reviewer 2.4.2 已解决哪些部分，仍缺什么证据")
    L.append("")
    L.append("**已处理：** 不再把重叠窗口当作独立观测；重采样单位是完整试次；四脑区同步抽取；会话内分块；主对象仍是池化比值差；家族用 Bonferroni。")
    L.append("**已处理：** 用已知真值的生成模型做误报率和覆盖率校准，校准集与验证集种子分离；把上一轮 14% 误报记为不通过而不是通过。")
    L.append("**未解决：** 没有在合理数据生成范围内获得足够校准的方法，因此不能可靠确认显著性。")
    L.append("**未解决：** 单受试者；电极抽样；会话间变异；微电极位置是否应在尖端；论文实际使用的三个会话文件；原始结果 CSV 缺失。")
    L.append("**未完成：** 按官方描述把微电极重分组到电极尖端，需要新估计约 112 个同束通道对（见偏离说明第 5 条）。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 9. 论文图 4c、正文和讨论应具体修改哪些表述")
    L.append("")
    L.append("- 删除窗口级 Welch 的 p 值、自由度（几十万到几百万窗口）和星号。")
    L.append("- 删除或不再作为主要论据的 p<0.001。")
    L.append("- 删除人群层面或“脑区一般性质”的表述。")
    L.append("- “海马延迟较高、颞上回较低”改为：在 sub-01 当前这组触点、这三个会话、池化加权下的描述性顺序；并写明去掉 AHL1 或改成通道对等权后顺序会变。")
    L.append("- 中位数只作描述；检验对象是均值（本轮连均值检验也未校准通过）。")
    L.append("- 图注写明：分布可以画窗口，但推断应以试次结构进行；本轮未能给出校准合格的显著性星号，建议不画星号。")
    L.append("- 讨论写明：共同参考、同束高相关、跨束近零相关、微电极位置可能在尖端，都会限制“区内延迟”的解释。")
    L.append("- 英文回复不要写 manuscript 已改，除非正文和图已经改完。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 10. 真实数据点估计（描述性，非正式显著性）")
    L.append("")
    L.append(md_table(ovn))
    L.append("")
    L.append("分会话、去一会话、零值、伪迹、权重、电极处理见 `realdata/results_all_methods_all_variants.csv`。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 11. 程序核验、种子、复用与耗时")
    L.append("")
    L.append(md_table(prog))
    L.append("")
    L.append("- 延迟：复用上一轮已核验的 9,000 个试次×通道对（4,572,000 窗口）；本轮新算 0。一致性验证 96/96 对来自上一轮。")
    L.append("- 种子：校准 `SeedSequence([20261031,1,…])`，验证 `[20261031,2,…]`，真实数据 `[20261031,3,…]`。")
    L.append(f"- 开始时估计：校准约 {time_est.get('E3_calibration_s', 0)/60:.1f} min；独立验证 bootstrap 约 {time_est.get('E4_validation_bootstrap_s', 0)/3600:.2f} h（规则：{time_est.get('validation_bootstrap_nsim_rule')}）。")
    if not timing.empty:
        L.append("")
        L.append(md_table(timing[["stage", "status", "seconds", "peak_rss_self_mb"]] if "peak_rss_self_mb" in timing.columns else timing, floatfmt=".1f"))
    L.append("")
    L.append("并行：6 个进程，每进程 BLAS 线程 1。断点续跑按情景×类型×批次保存。")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 12. 尚未解决 / 未完成")
    L.append("")
    L.append("1. 没有校准合格的主要推断方法。")
    L.append("2. 微电极是否应全部记为电极尖端位置（需作者确认）。")
    L.append("3. 按尖端重分组的延迟尚未重算。")
    L.append("4. 论文实际三个会话文件与原始 CSV 仍缺失。")
    L.append("5. ECL/LR 侧别、坐标单位、信号单位、SOZ 中的 AHR。")
    L.append("")
    L.append("这些缺口使“脑区差异显著”无法成立；描述性毫秒差和触点组合依赖仍然可以报告。")
    (OUT / "REPORT_CN.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    # ---------- English Methods ----------
    M = []
    M.append("# Methods paragraph (draft; Fig. 4c / Reviewer 2.4.2 calibration)")
    M.append("")
    M.append("This is a draft for a possible Methods revision. The manuscript text has not been edited in this repository.")
    M.append("")
    M.append("We re-examined the four-region comparisons of mean absolute delay in subject sub-01, sessions 01–03 (150 trials).")
    M.append("Delays were those already estimated with the original NeuroTD settings (W=1001 samples, hop H=500, rectangular window, discrete candidates ±500 samples, 32 kHz).")
    M.append("Channel mapping used the ElectricalSeries electrode references, not the first 32 rows of the electrode table.")
    M.append("The estimand was the pooled mean absolute delay, μ_r = Σ S_r / Σ N_r, where S and N are trial-level sums of non-zero absolute delays and counts; differences were Δ_AB = μ_A − μ_B.")
    M.append("Inference treated whole trials as the resampling units, kept the four regions paired, and resampled separately within each session.")
    M.append("We compared circular block bootstrap (percentile and studentized) and HAC variance estimators with fixed-b reference distributions.")
    M.append("False-positive rates and interval coverage were evaluated on independent calibration and validation simulation sets under global and partial nulls, including independence, AR(1) dependence, and designs matched to the observed trial-level residuals.")
    M.append("Method choice used only the calibration set and pre-specified pass/fail rules. In this analysis no method passed the core-scenario screen, so we do not report calibrated significance and treat the millisecond differences as descriptive for this subject, these contacts, and these sessions.")
    M.append("Electrode metadata were checked against the NWB files and the dataset descriptor (Boran et al., 2020, Sci Data). Conflicts between microelectrode channel labels and the described tip location of the microelectrodes are reported; laterality was not changed on the basis of naming or inferred millimetre distances.")
    (OUT / "methods_paragraph_EN.md").write_text("\n".join(M) + "\n", encoding="utf-8")

    # ---------- Reviewer response ----------
    R = []
    R.append("# Draft reply to Reviewer 2.4.2")
    R.append("")
    R.append("This draft states only work that has been completed. The manuscript text and Figure 4c have not been revised in the repository, so this reply does not claim that the paper has already been changed.")
    R.append("")
    R.append("We thank the reviewer for pointing out that overlapping windows are not independent observations. We agree that the original window-level Welch tests used an inappropriate denominator and that the associated p-values and asterisks are not valid.")
    R.append("")
    R.append("What we did. We reused the verified per-window delays for sub-01 sessions 01–03 (150 trials, 60 intra-region pairs). The comparisons remain the six pairwise differences of pooled mean absolute delay. We replaced window-level tests with trial-level procedures that keep all channel pairs and windows of a trial together and that resample within session. We then calibrated those procedures with simulation models whose true regional means are known, including independence, AR(1) dependence (φ=0.3 and the previously studied φ=0.6), designs matched to the observed trial-level residuals, and stress tests (longer memory, mild trend, heavy tails, a few outliers). Calibration and validation used separate random seeds. We also checked electrode metadata and raw waveforms for near-zero-delay pairs.")
    R.append("")
    R.append("What we found about Type I error. The previous circular block bootstrap with block length L=5 trials is anti-conservative when trials are AR(1) with φ=0.6 and n=50 trials per session. The earlier synthetic experiment (42/300 ≈ 14.0% rejections at a nominal 5% family level) is consistent with an analytic calculation: after demeaning, the expected circular-block variance is only about 57% of the true long-run variance, so standard errors are too small. The previous report marked that synthetic check as “passed”; that label was incorrect. Studentized block bootstrap and HAC/fixed-b intervals improve the error rate under independence and weaker dependence, but in our pre-specified core set—which includes φ=0.6 and the residual structure of these data—no method simultaneously kept family-wise error at the required level and maintained 95% coverage. Under the rule we wrote down before looking at the real-data p-values, we therefore do not designate a primary inferential method and we do not claim calibrated significance for any of the six comparisons.")
    R.append("")
    R.append("What this means for the six differences. The descriptive pooled differences remain on the order of 0.08–1.00 ms. We do not interpret the previous p<0.001 statements as valid, and we do not replace them with a new star rating. “Not having a reliable p-value” is not evidence that the delays are equal.")
    R.append("")
    R.append("Electrode composition. The “hippocampus” group in this subject has three contacts (AHL1, AHL2, PHR1). Removing AHL1 removes two of the three pairs and reverses the sign of the hippocampus-versus-MTG and hippocampus-versus-STG pooled differences. That is a change of contact set, not a demonstrated recording error. Several neighbouring pairs have a very high fraction of zero estimated delays; raw traces are not sample-wise identical, so we did not exclude them under the pre-specified quality rule. NWB location strings assign different regions to microelectrode channels that, according to the dataset descriptor, all sit near the shaft tip. We therefore cannot treat the four labels as verified regional populations.")
    R.append("")
    R.append("What we believe is resolved, and what is not. The specific objection—that overlapping windows were treated as independent—has been addressed by changing the inferential unit and by reporting a proper Type I error study. What remains missing is a procedure that is well calibrated for the dependence and session heterogeneity seen here, data from more than one subject, and confirmation of microelectrode locations from the data providers. Until those are available, we think the safe use of Figure 4c is descriptive.")
    R.append("")
    R.append("If the manuscript is revised, we would remove the window-level Welch p-values and asterisks, state the millisecond differences with their trial-level intervals as descriptive, and restrict the claim to this subject, these recorded contacts, and these three sessions. Those edits have not been made in the present files.")
    (OUT / "response_reviewer_2_4_2_EN.md").write_text("\n".join(R) + "\n", encoding="utf-8")

    if REP:
        REP.line("已写出 REPORT_CN.md、methods_paragraph_EN.md、response_reviewer_2_4_2_EN.md")
