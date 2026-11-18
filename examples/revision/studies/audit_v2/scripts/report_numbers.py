"""Builds the old-vs-new significance table and an auto-generated numeric summary (English)."""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

import infer as I
from common import CAL, ELE, OUT, PREV, REAL, REP_DIR, TAB, WQC, load_state, sha256_file


def md_table(df, floatfmt=None, index=False):
    """Markdown table without the tabulate package."""
    if df is None or len(df) == 0:
        return "*(Empty table)*"
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


def summary_verdicts():
    table = pd.read_csv(CAL / "validation_verdicts_by_method.csv")
    if table.verdict.map(str.isascii).all():
        return table
    # Each FWER scope contains five scenarios and three nulls. Unevaluable cells
    # were excluded from all three counts. Coverage has no resolution gate.
    counted = table.n_pass + table.n_fail + table.n_insufficient
    table["verdict"] = np.select(
        [(table.scope != "core_coverage95") & (counted < 15), table.n_fail > 0, table.n_insufficient > 0],
        ["not evaluable", "fail", "insufficient evidence"], default="pass")
    return table


def real_rows():
    from pipeline import VARIANT_TEXT, method_param

    table = pd.read_csv(REAL / "results_all_methods_all_variants.csv")
    original = table.pairs_removed_A.isna() & table.pairs_removed_B.isna()
    if (table.loc[original, "data_version"] == "original mapping").all():
        return table
    # The archived screen selected no method. Decisions for other historical
    # selections require their original calibration state and diagnostics.
    if load_state().get("selected_method") or table.is_selected_method.any():
        raise ValueError("Use the English analysis stages to rebuild decisions for a selected method.")
    table.loc[original, ["data_version", "electrode_handling"]] = ["original mapping", "All 60 within-region channel pairs"]
    pairs = pd.read_csv(ELE / "pair_classification.csv")
    def counts(subset):
        return tuple(subset.groupby("region").size().reindex(I.REGIONS, fill_value=0))

    excluded = set(load_state()["excluded_pairs"])
    keep = ~(pairs.label_i + "–" + pairs.label_j).isin(excluded)
    variants = [
        (keep, "evidence-based correction", f"Exclude pairs meeting the prespecified quality rule ({len(excluded)} pairs)"),
        (pairs.zero_fraction_all_trials < .70, "exploratory sensitivity", "Exclude pairs with ≥70% zero-delay windows"),
        (pairs.same_side_label, "exploratory sensitivity", "Pairs labelled ipsilateral only"),
        (~pairs.same_side_label, "exploratory sensitivity", "Pairs labelled contralateral only"),
        (pairs.same_bundle, "exploratory sensitivity", "Within-bundle pairs (same electrode) only"),
        (~pairs.same_bundle, "exploratory sensitivity", "Between-bundle pairs only"),
        (~pairs.involves_ECL, "exploratory sensitivity", "Exclude pairs involving ECL (conflicting laterality)"),
    ]
    contacts = set(pairs.label_i) | set(pairs.label_j)
    for label, group in table.loc[~original].groupby("electrode_handling"):
        contact = re.search(r"([A-Z]+[0-9]+)$", label)
        if contact and contact[1] in contacts:
            version, handling = "exploratory sensitivity", f"leave-one-contact-out: {contact[1]}"
        else:
            regional = pd.concat([group.set_index("region_A").pairs_A, group.set_index("region_B").pairs_B])
            if regional.groupby(level=0).nunique().ne(1).any():
                raise ValueError("Inconsistent regional pair counts for an archived electrode subset.")
            signature = tuple(regional.groupby(level=0).first().reindex(I.REGIONS))
            matches = [(kind, text) for mask, kind, text in variants if counts(pairs[mask]) == signature]
            if len(matches) != 1:
                raise ValueError(f"Unidentified or ambiguous electrode subset with regional counts {signature}.")
            version, handling = matches[0]
        table.loc[group.index, ["data_version", "electrode_handling"]] = [version, handling]
    table["analysis_role"] = table.analysis_variant.map(lambda v: VARIANT_TEXT[v][0])
    table["estimand"] = table.analysis_variant.map(lambda v: VARIANT_TEXT[v][1])
    table["method_param"] = table.method.map(method_param)
    table["decision"] = "insufficient evidence (no method passed the calibration screen)"
    table.loc[table.pairs_A.eq(0) | table.pairs_B.eq(0), "decision"] = "no data (no remaining pairs in one region)"
    diag = table.diagnostics.str.extract(r"lag-1 ACF\([^)]*\)=([^;]+);[^=]+=([^=]+)$")
    if diag.isna().any().any():
        raise ValueError("Unrecognized format for archived real-data diagnostics.")
    table["diagnostics"] = "lag-1 ACF(influence)=" + diag[0] + "; within-session trend t=" + diag[1]
    scope = summary_verdicts()
    scope = scope[scope.scope.eq("core")]
    labels = {m: "; ".join(f"α={r.alpha_family:g}:{r.verdict}" for r in rows.itertuples())
              for m, rows in scope.groupby("method")}
    table["calibration_scope"] = table.method.map(labels)
    return table


def metadata_table(path):
    """Render English annotations from the identified 2026-10-31 archive."""
    table = pd.read_csv(path)
    if path.name == "pair_metrics_summary.csv":
        selections = pd.read_csv(ELE / "pair_classification.csv").set_index("pair_index")
        zero = table.pair_index.map(selections.zero_fraction_all_trials)
        if zero.isna().any():
            raise ValueError("A waveform pair is missing from the electrode classification table.")
        table["selection_reason"] = np.select(
            [zero >= .70, table.region.eq("Hippocampus")],
            ["High zero-delay fraction (≥70%)", "Hippocampal pair"], default="Control pair")
        return table
    field, fingerprint = {
        "electrode_findings.csv": ("topic", "f397946c0ef5445f72865e89d65080c024a32296bc295491c1e3b7abac09130c"),
        "program_validation.csv": ("check", "acad2e064f63718af837f4994c5fb2ee27d32e1bd7ed616b5d3cc651d38b2b42"),
        "timing_summary.csv": ("stage", "8000d973a2e3d0bb4291aa765cfc3d00677d80aead2be65e0885bc22a9ab0408"),
    }[path.name]
    if str(table[field].iloc[0]).isascii():
        return table
    # These historical free-text annotations lack machine-readable labels.
    # Bind their translation to the verified input, never to arbitrary rows.
    if sha256_file(path) != fingerprint:
        raise ValueError(f"Unknown historical annotations in {path.name}. Regenerate this table with the English stages.")
    if field == "topic":
        findings = {
            "E1": ("Waveform-to-electrode mapping", "confirmed", "The 32 waveform columns reference rows 19–42 and 59–66 in all three sessions. Rows 0–18 are scalp EEG."),
            "E2": ("Micro and macro electrode rows", "conflicting evidence", "All 32 microwire rows are also among the 48 macroelectrode references. There are no separate microwire rows."),
            "E3": ("Regional labels within bundles", "conflicting evidence", "AHL1–2, AHL5 and AHL6–8 carry hippocampus, STG and MTG labels although the descriptor places the eight microwires near one tip."),
            "E4": ("Coordinate units and frame", "insufficient evidence", "The axes specify +x posterior, +y inferior and +z right. Adjacent macro contacts average 5.66 coordinate units versus 5 mm nominal spacing. Native units remain unconfirmed."),
            "E5": ("AHL1, AHL2 and PHR1", "conflicting evidence", "AHL1–AHL2 is ipsilateral within one bundle. The two AHL–PHR1 pairs cross shafts and hemispheres."),
            "E6": ("ECL laterality", "conflicting evidence", "ECL1–8 are labelled right despite left-side coordinates (z from −59.9 to −24.3)."),
            "E7": ("LR laterality", "conflicting evidence", "LR1–8 are labelled left but have positive z (22.1 to 61.4). LR occurs only in macroelectrode recordings."),
            "E8": ("Seizure-onset zone", "insufficient evidence", "The subject description names AHR and LR. AHR is absent from the depth-electrode list and only LR is flagged in the table."),
            "E9": ("Filter specification", "conflicting evidence", "The table specifies 0.5–1000 Hz. The descriptor reports a 0.5–5000 Hz passband and 32 kHz sampling."),
            "E10": ("Referencing", "confirmed", "Both the electrode metadata and the descriptor specify a common intracranial reference."),
            "E11": ("Signal units and conversion", "conflicting evidence", "The declared unit is volts with conversion 1.0. Quantization step × 32767 = 400.000, peak = 200.0000 and median RMS ≈6.3 (macro ≈53). The physical scale remains unconfirmed."),
            "E12": ("Contact identity across sessions", "confirmed in metadata", "All three sessions have the same electrode rows, indices, labels, locations and coordinates."),
            "E13": ("Microwire spectra and cross-bundle correlations", "insufficient evidence", "Median power fractions are 0.102 at 1–30 Hz and 0.760 at 30–3000 Hz. After 200 Hz low-pass filtering, between-bundle and micro–macro correlations are near zero; adjacent within-bundle correlations are 0.3–0.7 and peak at zero lag."),
        }
        return pd.DataFrame([(i, *findings[i]) for i in table.id], columns=["id", "topic", "category", "finding"])
    if field == "stage":
        from pipeline import STAGES
        names = dict(STAGES)
        order = ["s1", "s2", "s3", "s4", "s3", "s4", "s5", "s6", "s7", "s8"]
        table["stage"] = [names[key] for key in order]
        table.loc[4, "stage"] += " (post hoc)"
        table["status"] = "completed"
    else:
        table["check"] = [
            "Autocovariance sum vs. direct iteration", "Circular Bartlett vs. analytic bootstrap variance",
            "Circular Bartlett vs. 200,000 bootstrap draws (±2%)", "Identical regions", "Reversed contrast",
            "Trial with N=0", "Region with N=0 throughout", "Detrended linear signal", "Zero artifact weight",
            "Seed reproducibility", "Within-session resampling",
            "Fixed-b 95% critical values vs. asymptotic polynomial (n=50, ±15%)", "Simulation generator truth",
        ]
        table["detail"] = [
            "Maximum absolute difference 2.0e-13", "L=5, maximum absolute difference 8.5e-14",
            "Ratio 0.9987–1.0045", "Δ=0.0, p(M1)=1.000, p(M0)=1.000", "Δ=0.6913 vs. −0.6913",
            "z=0.0, pooled mean agrees with direct calculation", "Invalid contrasts [1, 3, 5]",
            "Maximum absolute residual 1.4e-14", "Agreement in all four regions", "Elementwise match",
            "2000 draws, L=5", "Finite vs. asymptotic at b=0.1, 0.2, 0.3, 0.5, 1.0: 2.242/2.261, 2.543/2.566, 2.859/2.874, 3.482/3.482, 4.781/4.813",
            "180/180",
        ]
    return table


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
    from pipeline import verdict_fwer

    disp = display_methods()
    vf = pd.read_csv(CAL / "validation_fwer.csv")
    cond = {}
    for m in disp:
        for al in I.ALPHAS:
            f = vf[(vf.method == m) & (vf.alpha_family == al) & vf.scenario.isin(COND_SCEN)]
            cond[(m, al)] = bool(len(f) == 12 and f.apply(verdict_fwer, axis=1).eq("pass").all())
    allr = real_rows()
    r = allr[allr.method.isin(disp)].copy()
    r["lag1"] = r.diagnostics.map(lambda s: float(re.search(r"ACF\(influence\)=([+-]?\d+\.\d+)", s).group(1)))
    lab = []
    for x in r.itertuples():
        if not x.test_performed or not np.isfinite(x.adjusted_p):
            lab.append("untested or missing")
            continue
        lv = [al for al in (0.001, 0.01, 0.05) if cond[(x.method, al)] and x.adjusted_p < al]
        if x.lag1 > 0.3:
            lab.append(f"insufficient conditional evidence (lag-1 influence ACF={x.lag1:+.2f} >0.3)")
        elif lv:
            lab.append(f"conditionally significant (adjusted p<{lv[0]:g})")
        elif x.adjusted_p < 0.05:
            lab.append("insufficient conditional evidence (this level failed conditional validation)")
        elif cond[(x.method, 0.05)]:
            lab.append("conditionally nonsignificant (compatible with a difference)")
        else:
            lab.append("insufficient conditional evidence")
    r["conditional_label"] = lab
    r["conditional_validation"] = r.method.map(lambda m: "; ".join(
        f"alpha={al:g}:{'pass' if cond[(m, al)] else 'fail'}" for al in I.ALPHAS))
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
    allr = real_rows()
    main = allr[(allr.analysis_variant == "main_pooled_nonzero") & (allr.session_scope == "pooled_3_sessions") &
                (allr.data_version == "original mapping")]
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
            d.update(selected_method="none", selected_decision="insufficient evidence (no method passed the calibration screen)")
        rows.append(d)
    cr, disp, _ = conditional()
    cm = cr[(cr.analysis_variant == "main_pooled_nonzero") & (cr.session_scope == "pooled_3_sessions") &
            (cr.data_version == "original mapping")]
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
    L.append("# Calibration and electrode verification: numerical results\n")
    L.append(f"Primary method under the prespecified rule: **{chosen or 'none'}**\n")
    sel = pd.read_csv(CAL / "method_selection.csv")
    L.append("## Method screen on the calibration set\n")
    L.append(md_table(sel, floatfmt=".4f"))
    vv = summary_verdicts()
    L.append("\n## Independent validation decisions for all methods\n")
    L.append(md_table(vv, floatfmt=".4f"))
    vf = pd.read_csv(CAL / "validation_fwer.csv")
    L.append("\n## Independent validation FWER at alpha=0.05\n")
    piv = vf[vf.alpha_family == 0.05].pivot_table(index="method", columns=["scenario", "null_type"], values="fwer")
    L.append(md_table(piv, floatfmt=".3f", index=True))
    L.append("\n## Previous and revised significance assessments\n")
    L.append(md_table(ovn, floatfmt=".4g"))
    allr = real_rows()
    cr = pd.read_csv(REAL / "conditional_results_display_methods.csv")
    L.append("\n## Exploratory conditional analysis (deviation 8)\n")
    L.append(md_table(pd.read_csv(CAL / "conditional_validation_display_methods.csv")))
    cols = ["method", "data_version", "electrode_handling", "analysis_variant", "session_scope", "direction",
            "difference_ms", "ci95_low", "ci95_high", "adjusted_p", "lag1", "conditional_label"]
    L.append(md_table(cr[cols], floatfmt=".4g"))
    if chosen:
        s = allr[(allr.method == chosen)]
        L.append("\n## Primary method across variants, scopes and contact handling\n")
        cols = ["data_version", "electrode_handling", "analysis_variant", "session_scope", "direction", "difference_ms",
                "ci95_low", "ci95_high", "adjusted_p", "decision", "pairs_A", "pairs_B", "limitations"]
        cols = [c for c in cols if c in s.columns]
        L.append(md_table(s[cols], floatfmt=".4g"))
    L.append("\n## Electrode findings\n")
    L.append(md_table(metadata_table(ELE / "electrode_findings.csv")))
    L.append("\n## Waveform checks\n")
    L.append(md_table(metadata_table(WQC / "pair_metrics_summary.csv"), floatfmt=".3f"))
    L.append("\n## Implementation checks\n")
    L.append(md_table(metadata_table(TAB / "program_validation.csv")))
    (REP_DIR / "results_numeric_summary.md").write_text("\n".join(L), encoding="utf-8")
    if REP:
        REP.line("Wrote reports/results_numeric_summary.md and realdata/old_vs_new_significance.csv")
