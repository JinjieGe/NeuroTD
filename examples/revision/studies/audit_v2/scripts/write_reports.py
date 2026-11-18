"""Write REPORT.md, English Methods, and the reviewer 2.4.2 draft from completed tables."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from common import CAL, ELE, LOGS, OUT, REAL, TAB, WQC, load_state
from report_numbers import md_table, metadata_table, old_vs_new, real_rows, summary_verdicts
from pipeline import verdict_fwer


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
    return f"{r.fwer:.3f}(CP [{r.fwer_cp_low:.3f}, {r.fwer_cp_high:.3f}], {verdict_fwer(r)})"


def write(REP=None):
    st = load_state()
    chosen = st.get("selected_method")
    fw = pd.read_csv(CAL / "validation_fwer.csv")
    vv = summary_verdicts()
    sel = pd.read_csv(CAL / "method_selection.csv")
    cfw = pd.read_csv(CAL / "calibration_fwer.csv")
    ovn = pd.read_csv(REAL / "old_vs_new_significance.csv")
    allr = real_rows()
    if not ovn.selected_decision.map(str.isascii).all():
        ovn = old_vs_new()
    cr = pd.read_csv(REAL / "conditional_results_display_methods.csv")
    findings = metadata_table(ELE / "electrode_findings.csv")
    wqc = metadata_table(WQC / "pair_metrics_summary.csv")
    theory = pd.read_csv(TAB / "old_method_theory_cbb_variance_ratio.csv")
    prog = metadata_table(TAB / "program_validation.csv")
    time_est = json.loads((LOGS / "time_estimate_at_start.json").read_text())
    timing = metadata_table(OUT / "timing_summary.csv") if (OUT / "timing_summary.csv").exists() else pd.DataFrame()
    disp = json.loads((CAL / "display_methods.json").read_text())
    boot_disp, ana_disp = disp["bootstrap"], disp["analytic"]
    condv = pd.read_csv(CAL / "conditional_validation_display_methods.csv")

    main = allr[(allr.analysis_variant == "main_pooled_nonzero") &
                (allr.session_scope == "pooled_3_sessions") &
                (allr.data_version == "original mapping")]
    loo = allr[(allr.analysis_variant == "main_pooled_nonzero") &
               (allr.session_scope == "pooled_3_sessions") &
               (allr.electrode_handling.fillna("").astype(str).str.startswith("leave-one-contact-out"))]

    def main_row(method, direction):
        q = main[(main.method == method) & (main.direction == direction)]
        return q.iloc[0] if len(q) else None

    directions = ["AMY − HIP", "AMY − MTG", "AMY − STG", "HIP − MTG", "HIP − STG", "MTG − STG"]
    # results table uses "AMY − HIP" or "Amygdala − Hippocampus" — check
    if "direction" in main.columns and len(main):
        directions = list(dict.fromkeys(main.direction.tolist()))

    n_pass = int(sel.passes_screen.sum())
    formal = "insufficient evidence (no method passed the calibration screen)"
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
            bits.append(f"{r.direction} {r.difference_ms:+.3f} ms (pairs removed: A {int(getattr(r, 'pairs_removed_A', -1))}/B {int(getattr(r, 'pairs_removed_B', -1))})")
        ahl1_note = "; ".join(bits[:6])

    # ---------- Audit report ----------
    L = []
    L.append("# Figure 4c calibration, electrode audit and reanalysis (Reviewer 2.4.2)")
    L.append("")
    L.append(f"Run directory: {OUT.name}/")
    L.append(f"Analysis date: 2026-10-31")
    L.append("This follow-up was designed after examining the preceding results. The protocol is in analysis_plan_calibration_electrodes.md.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 0. Summary")
    L.append("")
    L.append(f"1. **No primary method passed the prespecified screen.** Across the core calibration scenarios (A/B1/B2/C1/C2 × G/P1/P2), candidates had to keep the upper Clopper–Pearson bound on FWER at or below 0.075 at family alpha=0.05 and achieve at least 0.93 coverage for each nominal 95% interval. Number passing among 18 candidates: {n_pass}.")
    L.append(f"2. **Prespecified decision for all six contrasts: {formal}.** Equality has not been established. Pooled point estimates reproduce the earlier differences of about 0.08–1.00 ms.")
    L.append("3. **Finite-sample variance explains the approximately 14% false-positive rate for L=5, phi=0.6.** After demeaning, the expected circular block bootstrap variance is about 0.57 (L=5) or 0.24 (L=1) times the true long-run variance. Narrow intervals inflate rejections. The previous pass label for 42/300=14.0% was erroneous and invalidates its 12/12 pass count.")
    L.append("4. **Studentization and HAC/fixed-b improve calibration under independent and weakly correlated trials.** At phi=0.6 or stronger dependence, excessive rejection or low coverage persists. All stress scenarios (longer correlation, long memory, trend, heavy tails, outliers) are reported and excluded from selecting a method on the basis of significance.")
    L.append("5. **No comparison supports a formal p<0.001 claim.** At family alpha=0.001, B=1,999 bootstrap draws give insufficient resolution. Although some analytic methods pass at this level in individual scenarios, none passed the overall screen.")
    L.append("6. **The larger hippocampal delay depends on the contact set containing AHL1.** Dropping AHL1 removes two of three hippocampal pairs, leaving AHL2–PHR1. The analysis identifies a composition effect without establishing a contact error.")
    L.append("7. **Near-zero delays did not identify duplicate channels.** Across 15 checked trials, no pairs met the exclusion rule (≥99% identical samples or ≥50% constant, saturated or missing samples). Highly correlated pairs such as AL1–AL2 still have nonzero differences. Their removal is exploratory.")
    L.append("8. **Electrode metadata limit the regional interpretation of Figure 4c.** Microwires reference macro-contact rows. Wires from the same bundle receive different labels along the shaft. ECL laterality conflicts with the stated axes, and distances cannot be confirmed in individual anatomical millimetres.")
    L.append("9. **Whole-trial inference addresses overlapping windows. Calibrated significance remains unresolved.** Report descriptive differences and intervals for the one subject, these contacts and three sessions. Remove the original Welch stars and p<0.001 claims.")
    L.append("")
    L.append("The conditional analysis (protocol deviation 8) supplements the prespecified assessment. Displayed methods: "
             f"bootstrap={boot_disp} and analytic={ana_disp}. "
             "A conditional significance label requires independent validation across A/B1/C1/C2 (omitting B2). The formal decision remains insufficient evidence.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 1. Corrections to the previous analysis")
    L.append("")
    L.append("Sources: tables/previous_round_errata.csv and synthetic_validation/synthetic_validation_results.csv from the earlier run.")
    L.append("")
    L.append("| Issue | Treatment |")
    L.append("|---|---|")
    L.append(f"| At AR(1) phi=0.6 with 50 trials/session, a nominal 5% test rejected in 34.3% (L=1) and 14.0% (L=5) of simulations | Re-estimated in independent validation. Approximate single-test rejection rates: {t60_l1.approx_rejection_at_nominal_0_05_single_test:.3f} for L=1 and {t60_l5.approx_rejection_at_nominal_0_05_single_test:.3f} for L=5. |")
    L.append("| The pipeline recorded passed=True and reported 12/12 passed | The calibration check failed. Implementation correctness and error-rate control are assessed separately. |")
    L.append("| Dropping AHL1 changed HIP−MTG from +0.695 to −0.544 ms and HIP−STG from +1.002 to −0.237 ms | The leave-one-contact-out analysis identifies the two of three hippocampal pairs removed. |")
    L.append("| Equal pair weights changed AMY−STG from +0.225 to −0.313 ms | Reported separately as sensitivity D. |")
    L.append("| One replicate of variant F at L=1 failed to fit, leaving B=49,999 | Included in the errata. The M3 detrended HAC analysis retains rank-deficiency diagnostics. |")
    L.append("| Some sensitivity results met p<0.05 only | The errata identify 46 rows below 0.05 but above or equal to 0.001. |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 2. Why the earlier method yielded about 14% false positives")
    L.append("")
    L.append("The preceding primary test used a within-session circular block bootstrap of centred absolute errors (L=5, B=50,000).")
    L.append("For a demeaned series of n=50 trials, the expected circular block variance with a Bartlett kernel and block length L underestimates the true long-run variance under the examined AR(1) model.")
    L.append(f"At phi=0.6, expected bootstrap variance divided by true variance is {t60_l1.expected_cbb_var_over_true_var:.3f} for L=1 and {t60_l5.expected_cbb_var_over_true_var:.3f} for L=5.")
    L.append("The smaller standard error increases the standardized absolute difference. Approximate rejection rates for a single nominal 5% test are "
             f"{t60_l1.approx_rejection_at_nominal_0_05_single_test:.3f} and {t60_l5.approx_rejection_at_nominal_0_05_single_test:.3f}, ")
    L.append("consistent with 34.3% and 14.0% across 300 earlier simulations.")
    L.append("")
    L.append("Each session has 50 trials, and circular resampling treats trials 50 and 1 as neighbours. With six correlated comparisons, the familywise error can be higher still.")
    L.append("The earlier pass labels checked completion, seed reproducibility and shared trial indices across regions.")
    L.append("")
    L.append("Independent-validation FWER under the global null (alpha=0.05) for M0 at L=1 and L=5:")
    L.append("")
    L.append("| Scenario | M0 L=1 | M0 L=5 |")
    L.append("|---|---|---|")
    for s in ["A", "B1", "B2", "C1", "C2", "D1", "D2", "D3", "D4", "D5"]:
        L.append(f"| {s} | {_fwer(fw, 'M0_CBB_abs_L1', s)} | {_fwer(fw, 'M0_CBB_abs_L5', s)} |")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 3. Improvements and remaining calibration failures")
    L.append("")
    L.append("Candidates: M0 centred CBB (L=1/2/5/10), M2 studentized CBB at the same lengths, M1 HAC/fixed-b (b=0.1–1), and M3 detrended HAC/fixed-b.")
    L.append("Method screening used the calibration set. Performance was assessed independently. A/B1/B2/C1/C2 are core scenarios and D1–D5 are stress tests.")
    L.append("")
    L.append("### 3.1 Calibration screen")
    L.append("")
    L.append(md_table(sel[["method", "max_fwer_core", "max_fwer_cp_high_core", "worst_core",
                           "min_coverage_core", "passes_screen", "C1_mean_median_halfwidth"]], floatfmt=".4f"))
    L.append("")
    L.append("### 3.2 Independent validation decisions (core and stress scenarios)")
    L.append("")
    L.append(md_table(vv[vv.scope.isin(["core", "stress", "core_coverage95"])], floatfmt=".4f"))
    L.append("")
    L.append("### 3.3 Independent-validation FWER for display methods (alpha=0.05, global null)")
    L.append("")
    L.append("| Scenario | Earlier M0 L=5 | Bootstrap display method | Analytic display method |")
    L.append("|---|---|---|---|")
    for s in ["A", "B1", "B2", "C1", "C2", "D1", "D2", "D3", "D4", "D5"]:
        L.append(f"| {s} | {_fwer(fw, 'M0_CBB_abs_L5', s)} | {_fwer(fw, boot_disp, s)} | {_fwer(fw, ana_disp, s)} |")
    L.append("")
    L.append("With independence or short correlation (A/B1) and near-observed residual correlation (C1/C2), studentization and HAC with larger b generally approached the nominal level more closely than M0 at L=1 or 5.")
    L.append("The core requirements failed at phi=0.6 (B2) and under longer dependence or approximate long memory. Interval coverage also fell below 0.93.")
    L.append("All stress-test failures are included.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 4. Decisions for the six contrasts")
    L.append("")
    L.append("Primary estimand: regional pooled mean μ_r=ΣS/ΣN over nonzero windows, 150 trials, three sessions and the original 60 within-region pairs.")
    L.append("")
    rows = []
    for dname in directions:
        r0 = main_row(boot_disp, dname)
        r1 = main_row(ana_disp, dname)
        c0 = cr[(cr.method == boot_disp) & (cr.direction == dname) &
                (cr.analysis_variant == "main_pooled_nonzero") &
                (cr.session_scope == "pooled_3_sessions") &
                (cr.data_version == "original mapping")]
        o = _row(ovn, comparison=dname) if "comparison" in ovn.columns else None
        if o is None and len(ovn):
            # try matching first column
            o = ovn.iloc[directions.index(dname)] if dname in directions and directions.index(dname) < len(ovn) else None
        rows.append({
            "Comparison": dname,
            "Δ (ms)": None if r0 is None else f"{r0.difference_ms:+.3f}",
            "Bootstrap 95% CI (display)": None if r0 is None else f"[{r0.ci95_low:+.3f}, {r0.ci95_high:+.3f}]",
            "Analytic 95% CI (display)": None if r1 is None else f"[{r1.ci95_low:+.3f}, {r1.ci95_high:+.3f}]",
            "Prespecified decision": formal,
            "Conditional label (exploratory)": "" if not len(c0) else c0.iloc[0].conditional_label,
        })
    L.append(md_table(pd.DataFrame(rows)))
    L.append("")
    L.append("**All six contrasts had insufficient evidence under the prespecified rule.** The analysis establishes neither reliable significance nor equality.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 5. Significance thresholds supported by the analysis")
    L.append("")
    L.append("- Under the prespecified rule, no comparison supports a formal claim at 0.05, 0.01 or 0.001.")
    L.append("- Bootstrap calibration is unevaluable at family level 0.001. B=1,999 gives a minimum p-value of 0.0005, above 0.001/6. This yields no estimate of false positives at that threshold.")
    L.append("- Conditional passes at 0.05 or 0.01 under A/B1/C1/C2 do not generalize to phi=0.6 or to the population. Labels are in realdata/conditional_results_display_methods.csv.")
    L.append("- The previous adjusted p=0.00012 is a resolution limit at B=50,000 and does not support significance at 0.001 for an uncalibrated method.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 6. Dependence of the hippocampal estimate on AHL1")
    L.append("")
    L.append("The hippocampal contact set is AHL1, AHL2 and PHR1. Removing AHL1 drops AHL1–AHL2 and AHL1–PHR1 and leaves AHL2–PHR1.")
    L.append("Previous pooled differences changed from +0.695 to −0.544 ms for HIP−MTG and from +1.002 to −0.237 ms for HIP−STG.")
    if ahl1_note:
        L.append(f"Leave-one-contact-out results for {chosen or boot_disp}: {ahl1_note}.")
    L.append("This does not establish an AHL1 recording error. Two pairs cross shafts and hemispheres (AHL left, PHR right). The descriptor locates AHL3–AHL8 near the same tip as AHL1/AHL2 despite their STG/MTG labels.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 7. Electrode labels and near-zero pairs")
    L.append("")
    L.append("### 7.1 Graded electrode findings")
    L.append("")
    L.append(md_table(findings[["id", "topic", "category"]]))
    L.append("")
    L.append("Confirmed: ElectricalSeries references map waveform columns to table rows, the reference is intracranial and shared, and metadata match across sessions.")
    L.append("Conflicting evidence: macro and micro channels share electrode rows, micro labels vary along shafts, the AHL1/AHL2/PHR1 group crosses hemispheres, ECL/LR labels conflict with axes, filter descriptions disagree and declared volts conflict with a numerical scale resembling ±200 full-scale int16.")
    L.append("Unresolved: coordinate units and reference frame, the AHR SOZ designation, microwire spectra and near-zero correlations across bundles.")
    L.append("Laterality was preserved pending independent evidence. See electrodes/questions_for_authors.md.")
    L.append("")
    L.append("### 7.2 Near-zero-delay pairs")
    L.append("")
    L.append(md_table(wqc[["pair", "region", "selection_reason", "identical_fraction_median",
                           "r_zero_lag_median", "diff_rms_ratio_median", "window_zero_fraction_mean",
                           "exclude_in_evidence_corrected"]], floatfmt=".3f"))
    L.append("")
    L.append("No pair met the prespecified exclusion rule. High correlation and frequently zero delays alone do not establish duplication, crosstalk or hardware failure.")
    L.append("In post hoc mismatched-trial checks, zero-delay fractions for AL1–AL2, PHR6–PHR7 and ECL5–ECL6 dropped from >95% to about zero, and mean nonzero delays rose to about 7–9 ms. The near-zero values depend on similar waveforms recorded simultaneously. The signals are not samplewise copies.")
    L.append("The quality-rule correction retains all pairs. Removing near-zero pairs and restricting by side or bundle remain exploratory sensitivities.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 8. Status of Reviewer 2.4.2 in this audit")
    L.append("")
    L.append("**Addressed:** Whole trials are the resampling units. The four regions share indices, blocks stay within sessions, the estimand remains a pooled ratio contrast and the family uses Bonferroni adjustment.")
    L.append("**Addressed:** Separate seeds for calibration and validation test false-positive rates and coverage under known truth. The earlier 14% false-positive result is recorded as a failure.")
    L.append("**Unresolved:** Under the plausible generators examined here no candidate achieved adequate calibration for confirmatory significance.")
    L.append("**Unresolved:** Single-subject scope, contact sampling, between-session variability, microwire tip localization, identity of the three submission input files and missing original result CSV.")
    L.append("**Pending:** Grouping by tip position as described in the dataset would require estimates for approximately 112 within-bundle pairs (deviation 5).")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 9. Implications for Figure 4c, Results and Discussion")
    L.append("")
    L.append("- Remove the window-level Welch p-values, degrees of freedom based on hundreds of thousands to millions of windows, and significance stars.")
    L.append("- Remove p<0.001 as primary evidence.")
    L.append("- Restrict population-level and region-wide claims.")
    L.append("- Present the higher hippocampal and lower STG estimates as descriptive of sub-01, these contacts, three sessions and pooled weights. Show how ordering changes after dropping AHL1 or weighting pairs equally.")
    L.append("- Use medians descriptively. The inference target is a mean, and this audit did not establish calibration for that target.")
    L.append("- In the caption, explain that the plots show windows while inference uses trials. Omit significance stars unsupported by the calibration.")
    L.append("- Discuss the common reference, high correlation within bundles, near-zero correlation between bundles and possible tip localization when interpreting within-region delays.")
    L.append("- Describe manuscript and figure changes in the response once they are in place.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 10. Real-data estimates (descriptive)")
    L.append("")
    L.append(md_table(ovn))
    L.append("")
    L.append("See realdata/results_all_methods_all_variants.csv for session-specific, leave-one-session-out and sensitivity results for zeros, artifacts, weights and contacts.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 11. Implementation checks, seeds, reuse and timing")
    L.append("")
    L.append(md_table(prog))
    L.append("")
    L.append("- Reused 9,000 verified trial-by-pair delay estimates (4,572,000 windows). The earlier raw-data verification matched 96/96 sampled combinations. No delays were newly estimated in this audit.")
    L.append("- SeedSequence prefixes: [20261031,1,…] for calibration, [20261031,2,…] for validation, and [20261031,3,…] for real data.")
    L.append(f"- Initial estimate: calibration {time_est.get('E3_calibration_s', 0)/60:.1f} min, independent bootstrap validation {time_est.get('E4_validation_bootstrap_s', 0)/3600:.2f} h (planned validation replicates: {st['val_nsim_boot']:,}).")
    if not timing.empty:
        L.append("")
        L.append(md_table(timing[["stage", "status", "seconds", "peak_rss_self_mb"]] if "peak_rss_self_mb" in timing.columns else timing, floatfmt=".1f"))
    L.append("")
    L.append("Six processes ran with one BLAS thread each. Checkpoints are saved by scenario, null type and batch.")
    L.append("")
    L.append("---")
    L.append("")
    L.append("## 12. Open items")
    L.append("")
    L.append("1. No primary method met the calibration criteria in this audit.")
    L.append("2. Author confirmation is needed for assigning microwires to electrode tip positions.")
    L.append("3. Delays have not been recomputed under tip-based regional assignment.")
    L.append("4. The three submission input files and original result CSV remain unidentified.")
    L.append("5. ECL/LR laterality, coordinate units, signal units and the AHR SOZ label need clarification.")
    L.append("")
    L.append("These issues limit formal inference about regional differences. Descriptive differences in milliseconds and their dependence on the contact set can still be reported.")
    (OUT / "REPORT.md").write_text("\n".join(L) + "\n", encoding="utf-8")

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
        REP.line("Wrote REPORT.md, methods_paragraph_EN.md and response_reviewer_2_4_2_EN.md")
