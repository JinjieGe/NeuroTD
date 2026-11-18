from pathlib import Path
import json,hashlib,html
import pandas as pd
import numpy as np
from run_analysis import ROOT,save_json,provenance

r=pd.read_csv(ROOT/'outputs/real_results.csv');v=pd.read_csv(ROOT/'outputs/final_validation.csv')
s=pd.read_csv(ROOT/'outputs/sensitivity.csv')
assert len(v)==90 and v.n.eq(20000).all()
assert len(list((ROOT/'checkpoints').glob('final_*.json')))==30
assert v[v.alpha.eq(.05)].gate.all() and v[v.alpha.eq(.01)].gate.all()
assert (v[v.alpha.eq(.05)].coverage_ci_low>=.94).all()
for p,h in json.loads((ROOT/'provenance_before.json').read_text()).items():
    assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h
q=r[r.method=='session_jackknife_t2'].set_index('comparison')
order=[('HIP-AMY','AMY-HIP',-1),('HIP-MTG','HIP-MTG',1),('HIP-STG','HIP-STG',1),
       ('MTG-AMY','AMY-MTG',-1),('AMY-STG','AMY-STG',1),('MTG-STG','MTG-STG',1)]
rows=[]
for name,k,sgn in order:
    a=q.loc[k];lo=a.lower_simultaneous95_ms;hi=a.upper_simultaneous95_ms
    if sgn<0:lo,hi=-hi,-lo
    rows.append(dict(comparison=name,estimate_ms=sgn*a.estimate_ms,p_bonferroni=a.p_family,
        lower_simultaneous95_ms=lo,upper_simultaneous95_ms=hi,df=2,
        conditional_significant_05=bool(a.p_family<.05)))
pd.DataFrame(rows).to_csv(ROOT/'outputs/recommended_results.csv',index=False)
table='\n'.join(f"{x['comparison']:9s} {x['estimate_ms']:+.6f} ms  P_adj={x['p_bonferroni']:.6f}  simultaneous 95% interval [{x['lower_simultaneous95_ms']:.6f}, {x['upper_simultaneous95_ms']:.6f}]" for x in rows)
summary=v.groupby('alpha').agg(cases=('gate','size'),passed=('gate','sum'),max_fwer=('fwer','max'),
    max_upper=('fwer_ci_high','max'),min_coverage=('simultaneous_coverage','min'),min_coverage_lower=('coverage_ci_low','min')).reset_index()
summary.to_csv(ROOT/'outputs/final_validation_summary.csv',index=False)
ver=json.loads((ROOT/'outputs/verification.json').read_text())
ver.update(final_validation_records=600000,total_formal_simulation_datasets=1440000,
    final_validation_rows_checked=90,final_primary_05_gate_passed=True,final_primary_01_gate_passed=True,
    final_primary_001_all_cases_passed=False,final_simultaneous95_coverage_gate_passed=True,
    all_original_source_hashes_still_match=True)
save_json(ROOT/'outputs/final_verification.json',ver)

report=f'''NeuroTD Reviewer 2.4.2: historical statistical reanalysis
2026-11-08

Findings
The November 8 audit retained the pooled contrast in nonzero absolute delays and estimated its standard error by deleting one session at a time. Six two-sided contrasts used a small-sample t(2) reference and Bonferroni adjustment. The procedure met the operational criteria at the 5% and 1% family levels in all 30 configurations of the independent validation. In the observed data, the three contrasts between the HIP-labelled contacts and each other group reached the 5% family level under these assumptions. The remaining three did not. These results do not support the former blanket p<0.001 statement. The small sample cannot establish session independence or support general claims about regional physiology, the population or neural conduction.

1. Data and simulation design
The audit used sub-01, three sessions, 150 trials, the original 60 channel pairs and 4,572,000 windows, including 4,108,812 nonzero estimates. Inputs were trial-by-label-group sums of nonzero absolute delays (S) and counts (N). Delay estimation was not rerun. Independently aggregating the 9,000 trial/pair records reproduced all 600 trial/group records. Pooled group means were AMY 8.898783, HIP 9.676558, MTG 8.981442 and STG 8.674186 ms. These names preserve source labels; microwire anatomy remained unverified. Before/after SHA-256 checks confirmed that the original inputs, estimator, E and previous experimental sources were unchanged.

The 1,440,000 simulated datasets comprised:
- 200,000 for the finite-model reference envelope (20,000 for each of 10 scenarios).
- 600,000 for comparing five procedures (10 scenarios × 3 null configurations × 20,000).
- 40,000 in two additional challenges (20,000 each).
- 600,000 independent datasets for evaluating the selected method after comparison.
A further 10,000 pilot datasets tested execution and measured runtime. Each dataset represented four group summaries for each of 150 simulated trials. The simulations reused delay estimates rather than generating new raw neural recordings. The first formal run took about 87.7 s and the final evaluation about one minute, using vectorized, small-batch, single-process calculations. The original models and seeds were retained. New simulations used a separately frozen seed, 202611080242.

2. Session-level uncertainty
Earlier short-block procedures primarily estimated trial-level uncertainty within fixed sessions and could miss longer autocorrelation. This audit deleted complete sessions, including all their trials, pairs and windows. Let d denote the pooled contrast and d_(-s) the contrast after omitting session s.
SE_jack = sqrt[(2/3) sum_s(d_(-s) - mean_s d_(-s))^2].
T = d / SE_jack. P_adj = min(1, 6 × 2 × Pr[t_2 >= abs(T)]).
The simultaneous 95% interval is d ± t_(2, 1 - .05/12) × SE_jack.
The reference df of 2 comes from three sessions minus one. The pooled point estimator is unchanged; its sampling interpretation now concerns session-level variation. This calculation does not estimate effective window degrees of freedom or replace the df of the original Welch test.
The reference requires sufficiently independent session errors, an appropriate sampling interpretation for these recording conditions, and a reasonable t approximation for session summaries. Three sessions provide limited information about these assumptions. General small-sample exactness of the ratio jackknife was not established. Ibragimov and Muller motivate the equal-session t sensitivity analysis; their theory does not automatically justify the unequally weighted ratio jackknife.

3. Real-data contrasts
All p-values below are two-sided and Bonferroni-adjusted over six comparisons. Intervals are simultaneous 95% intervals under the same reference model.
{table}

HIP–MTG also meets the 1% family level. HIP–STG gives approximately 0.010195, above 0.01 despite rounding. None of the six reaches p<0.001. A nonsignificant result does not establish equivalence.

4. Independent calibration checks
The prespecified operational criterion required the upper limit of the 95% Clopper–Pearson interval for familywise error to be at most 1.25 × nominal alpha. This criterion allows limited deviation. Each interval pertains to its configuration; it gives no simultaneous guarantee over all 30 configurations.
Results from the final 600,000 datasets:
- alpha=.05: 30/30 passed. Maximum observed FWER 4.875%; maximum upper limit 5.183%.
- alpha=.01: 30/30 passed. Maximum observed FWER 1.045%; maximum upper limit 1.196%.
- alpha=.001: 27/30 passed. A-G, B1-G and B2-G missed the upper-limit criterion. Maximum observed FWER 0.145%; maximum upper limit 0.208%. Finite simulation uncertainty contributes to these upper limits; the audit did not validate this family level in every scenario.
- Minimum simultaneous 95% coverage was 95.125%. Its lowest Clopper–Pearson lower bound was 94.817%, above the prespecified 94% threshold.
The designs included independence, AR coefficients .3/.6, observed-data approximations, random signal-correlated denominators, AR=.85, slow components, trends, heavy tails, heteroscedasticity and outlying trials. Each scenario had a global and two partial nulls. These results condition on fixed generating parameters; parameters estimated from the observed table carry additional uncertainty.

5. Alternative methods
The equal-session t(2) test supported the same three HIP contrasts at .05, with adjusted p-values of about .0290, .00716 and .01033. It targets equal-weight session means. In scenarios C and D the population contrast for this target can be nonzero. A blank FWER field there signifies no applicable true null, and simultaneous coverage was evaluated against that method's own target.
HAC b=1 with a normal reference was a diagnostic counterexample. This bandwidth/reference combination was poorly calibrated and did not reproduce the earlier fixed-b test. Printed zeros reflected numerical tail underflow. HAC b=1 with a t(2) reference was conservative in the core scenarios, but global-null FWER reached about 8.275% in the earlier slow-component stress case.
The finite-model envelope calibrated the maximum standardized error across six contrasts against ten earlier models, with a fixed conservative factor of 1.15. Its maximum FWER in those scenarios was about 2.505%. Small p-values for HIP contrasts do not validate the method outside the reference models. Parameters came from the real data and the envelope did not maximize over the entire nuisance parameter space. The audit retained these comparisons without selecting the smallest p-value.

6. Additional challenges
E1 used within-session AR=.97 with independent sessions. FWER was 4.845% for the session jackknife, 21.305% for HAC-t(2), and 9.735% for the envelope.
E2 added a random offset per group shared across all three sessions (SD=.2 ms), defined as measurement error. FWER rose to 52.065% for the session jackknife and 98.335% for the envelope. Deleting individual sessions cannot estimate an offset shared by every session. These simulated rates do not measure the real-data false-positive rate or establish that the recordings contain this offset. Defining a persistent offset as part of the participant's true conditional effect would change the null hypothesis. The challenge illustrates the identifiability limit between a shared measurement bias and an effect and restricts inference to the stated dependence assumptions.

7. Sensitivity and generalizability
Excluding 16 artifact-flagged trials retained the direction and .05 classification of the three HIP contrasts (adjusted p about .0197, .00766 and .00744). Including zeros changed both the target and the p-values; these alternatives cannot substitute for the primary result.
Removing AHL1 shifted HIP–MTG from +0.695 to −0.544 ms and HIP–STG from +1.002 to −0.237 ms. This also removes two of the three HIP pairs, documenting sensitivity to composition without determining which contact label is correct. These variants did not receive separate calibration banks. Their p-values are sensitivity results, and the labels and contact selection limit biological interpretation.
Exact sign flips across just three sessions gave unadjusted p=.25 or .5, all 1 after correction for six contrasts. The t results depend on stronger distributional assumptions. The method recommendation was retrospective: the historical findings and observed p-values were already known. The independent simulation evaluation followed selection and does not turn the real-data reanalysis into a prospective biological validation.

8. Historical revision recommendations
Withdraw the original window-level Welch p<.001 and uniform significance stars. Report the conditional df=2 session-level analysis and its independent validation. The three HIP contrasts reached the .05 family level; the others did not. Explain the uncertain anatomy, single participant, three sessions, shared errors and contact composition. If session independence cannot be justified, keep the main-text findings descriptive and present the conditional p-values as supplementary sensitivity analyses. This provides a statistical response under explicit assumptions; residual dependence and regional neurophysiology remain open. The manuscript and online response document were not edited by this audit. The output includes an English response draft and proposed Methods, Results and Discussion revisions.

9. Reproducibility
All 600 trial/group entries matched independent aggregation of the 9,000 trial/pair rows. The fast HAC identity and the previous explicit covariance formula differed by less than 8e−19. SciPy independently reproduced the session t statistic; independent formulas verified the jackknife and t(2) tail probabilities. Replaying the first 1,000 samples for three reference scenarios matched bit for bit. Empirical tail counts matched direct counts. The 90 final validation rows, 30 checkpoints and key file hashes passed checks. Three PNG figures were visually checked in the historical run. The audit directory records the scripts, frozen plans, seeds, results, checkpoints and hashes.

References
Ibragimov & Muller (2010), t-Statistic Based Correlation and Heterogeneity Robust Inference.
https://www.princeton.edu/~umueller/tstat.pdf
Dufour (2006), Monte Carlo tests with nuisance parameters. This distinguishes a finite reference grid from calibration over the full nuisance parameter space; the audit does not claim all exactness conditions hold.
https://jeanmariedufour.research.mcgill.ca/Dufour_2006_JE_MCT.pdf
'''
(ROOT/'analysis_EN.txt').write_text(report)

en='''Draft response to Reviewer 2.4.2
Status: supplementary analyses completed; manuscript edits have not yet been applied. Add page/line references and change future tense only after those edits are made.

We agree that the original window-level analysis did not adequately account for dependence. Overlapping delay estimates are not independent replicates, and neither Welch's test nor Bonferroni adjustment resolves this issue. We therefore do not retain the original window-level p<0.001 values as formal inferential evidence.

Keeping the NeuroTD estimator and estimated delays unchanged, we reanalysed the same participant's three sessions and 150 trials. We retained the original differences in pooled nonzero absolute-delay means, estimated standard errors by deleting one entire session at a time, and used a Student-t reference with two degrees of freedom, with Bonferroni adjustment for six two-sided contrasts. The reference degrees of freedom arise from the three sessions, rather than the number of windows. This analysis requires a defensible session-level sampling interpretation and sufficiently independent errors between sessions; it is not an estimate of a universal effective window count.

After comparing candidate procedures and documenting the post-hoc recommendation, we evaluated this procedure using a further 600,000 synthetic datasets that had not been used in the method comparison. The final evaluation included ten legacy generating scenarios and three global/partial-null configurations per scenario. All 30 configurations met the prespecified operational criterion at family levels of 0.05 and 0.01: the upper 95% binomial confidence bound for family-wise error was no greater than 1.25 times the nominal level. At the 0.05 level, the maximum observed family-wise error was 4.875%, the maximum upper confidence bound was 5.183%, and the minimum simultaneous 95% interval coverage was 95.125%. These are model-conditional validation results, not a proof of exact error control under arbitrary real-data dependence. The 0.001 level did not meet the criterion in all configurations.

Under this conditional analysis, the original HIP-labelled group exceeded the AMY-, MTG-, and STG-labelled groups by 0.778, 0.695, and 1.002 ms, with adjusted p-values of 0.0290, 0.00726, and 0.0102, respectively. The remaining three contrasts did not reach the 0.05 family level. A separately labelled equal-session t analysis yielded the same three rejections, although its aggregation target differs slightly. We will replace the original blanket p<0.001 statement and associated significance annotations with this qualified account.

We also explicitly tested a failure boundary: shared random measurement offsets across sessions can invalidate session-level inference. Accordingly, the new p-values will be presented as assumption-dependent supplementary evidence, not as distribution-free evidence or population-level inference. Microelectrode anatomical labels remain unverified, and some contrasts are sensitive to contact composition. Interpretation will therefore be restricted to this participant's recorded contacts and sessions. If session independence cannot be adequately justified, the main-text presentation will remain descriptive and the conditional inference will be confined to the supplement.
'''
(ROOT/'response_EN.txt').write_text(en)
methods='''Proposed manuscript replacement text — not applied to manuscript

Methods
Window estimates were not treated as independent replicates. For each original label-defined group, the descriptive estimator was the sum of nonzero absolute delays divided by the number of nonzero estimates across the three sessions. For each pairwise contrast d, we recomputed this ratio difference after deleting each complete session, giving d_(-s), s=1,2,3. The session-jackknife standard error was [ (2/3) sum_s {d_(-s)-mean_s d_(-s)}^2 ]^(1/2). We report conditional two-sided Student-t reference p-values with two degrees of freedom and Bonferroni adjustment for six comparisons, together with corresponding simultaneous 95% intervals. These are supplementary, assumption-dependent analyses: independence of session-level errors and an appropriate session-level sampling interpretation are required. They do not establish a corrected number of independent windows or generalize to multiple participants. The method recommendation was retrospective; a separate simulation set evaluated it after the comparison of candidate methods.

Results
In this single participant, the pooled nonzero absolute-delay differences for HIP-labelled versus AMY-, MTG-, and STG-labelled contacts were 0.778, 0.695, and 1.002 ms. Conditional session-jackknife adjusted p-values were 0.0290, 0.00726, and 0.0102, respectively. Corresponding simultaneous 95% intervals were [0.187,1.369], [0.432,0.958], and [0.552,1.453] ms. The other three contrasts did not reach the 0.05 family level. The original window-level Welch p-values are not used for inference. The labels are retained for provenance and do not indicate independently verified microelectrode anatomical locations.

Discussion
The supplementary analysis reduces the inferential replication level from individual windows to complete sessions and substantially weakens the original significance claims. Simulation evaluation supports the reported procedure within the tested generating scenarios, but does not establish validity under arbitrary dependence. In particular, shared random measurement offsets across sessions are not captured by deleting sessions. With only three sessions from one participant, the assumed independence and distributional approximation cannot be thoroughly verified. Contact-composition sensitivity and unresolved anatomical mapping further limit interpretation. The estimates should be understood as alignment-delay summaries of the recorded signals, not as established regional transmission times or population-level physiological differences.

Figure caption addition
Points show pooled nonzero absolute-delay contrasts; crosses show individual-session contrasts. Intervals are conditional simultaneous 95% session-jackknife t(2) intervals adjusted for six contrasts. Three sessions belong to one participant. Original anatomical labels are provisional. No individual window is counted as an independent replicate in this supplementary inference.

Implementation status
These paragraphs are proposed replacements only. The original manuscript and online response document have not been edited.
'''
(ROOT/'manuscript_replacement_EN.txt').write_text(methods)

# Standalone human-readable report; no external assets needed beyond relative local figures.
body=html.escape(report).replace('\n','<br>')
doc=f'''<!doctype html><html lang="en"><meta charset="utf-8"><title>NeuroTD 2.4.2 historical reanalysis</title>
<style>body{{max-width:1100px;margin:40px auto;padding:0 24px;font:16px/1.75 system-ui;color:#203040}}img{{width:100%;margin:20px 0}}pre{{white-space:pre-wrap}}h1{{font-size:28px}}</style>
<h1>NeuroTD 2.4.2 historical reanalysis</h1><p>2026-11-08 · One participant, three sessions · Conditional inference</p>
<img src="outputs/figure1_contrasts.png" alt="Regional contrasts and session-level intervals">
<div>{body}</div><img src="outputs/figure2_calibration.png" alt="Initial validation and model challenges"><img src="outputs/figure3_sensitivity.png" alt="Sensitivity comparisons"></html>'''
(ROOT/'analysis_EN.html').write_text(doc)
save_json(ROOT/'outputs/delivery_facts.json',dict(recommended_results=rows,validation=summary.to_dict('records'),
    total_formal_simulation_datasets=1440000,window_estimator_rerun=False,original_code_modified=False,manuscript_modified=False,
    inference_scope='conditional_on_session_independence_and_other_stated_assumptions',
    unconditional_effective_df_correction_proven=False))
print(table)
