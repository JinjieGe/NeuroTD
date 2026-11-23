# Reviewer comments and supporting code

Numbering follows the response document. Comment 2.6 corresponds to Reviewer 2's fifth major comment. The revision bundle excludes the reliability analyses for Comment 2.5. The [run guide](revision-running.md) shows how to verify Table 1, rebuild Tables S1–S9 and S11–S12, and export Supplementary Data S1.

| Comment | Analysis or entry point | Evidence and scope |
|---|---|---|
| 1.1, 1.2 | `src/neurotd/time_varying_shift.py`, `score_uncertainty/`, `figures_13/` | Relationship between MSE and correlation under stated conditions, and comparisons against known delays. Performance applies to the settings and baseline implementations tested. |
| 1.3 | `examples/`, `data/`, `docs/`, `tests/` | Separates the reusable package, examples, data, and documentation. |
| 2.2 | `score_uncertainty/audit.py`, `score_uncertainty/summarize.py` | Compares minimum-E and oracle windows in 1,200 noisy signal pairs with fixed clean templates. Includes high-noise failures on repeated waveforms. |
| 2.3 | `score_uncertainty/`, `figures_23/build_figure.py`, `search_range/` | Candidate scores, selection frequencies, and search-range sensitivity. Assess accuracy separately against known delays. |
| 2.4.1 | `score_uncertainty/` | Estimates residual covariance and the SD and Monte Carlo SE of E across independent signal pairs. The score retains its original window count. |
| 2.4.2 | `session_inference/code/calibrated_welch.py` | Six regional comparisons from trial summaries, with calibrated SEs, estimated degrees of freedom, and Bonferroni correction (S6–S7). Inference is conditional on one participant's three sessions. Includes out-of-library stress tests. |
| 2.4.3 | `dynamic_fixed/summarize_matched_controls.py`, `dynamic_fixed/`, `dynamic_selection/` | Applies Holm correction to 72 endpoints per simulated panel and to the 72 observed-record endpoints after combining the three fitted reference processes. Reports familywise false positives, record-level detection, and tracking error for matched interference controls. The proportion of artifacts in real recordings remains unknown. |
| 2.6 | `examples/ieeg/`, `examples/patchseq/`, `dataset_reporting/`, `figures_26/` | Original workflows, session and trial counts, and processed sweeps. File inventories characterize their inputs. |
| 2.7.1 | `snr_initial/`, `snr_extension/` | 253,440 noisy pairs from 36 templates, eight known shifts, eleven noise levels, two noise types, two channel configurations, and 20 repetitions. SNR refers to added noise. Templates came from one participant and three sessions. |
| 2.7.2 | iEEG pairing code, `audit_v3/` | Electrode composition and local matching diagnostics. Matching between distinct real events requires separate physiological evidence. |
| 2.7.3 | iEEG signed outputs and absolute-delay summaries | Pair order determines the sign. Within-region magnitudes do not establish directional transmission. |
| 2.8.1 | `search_range/run_search_range_ablation.py` | Compares half-ranges W/4 and W/2 at native and shared centers. |
| 2.8.2 | `dynamic_fixed/` and the iEEG/Patch-seq entry points | Distinguishes within-window weighting, reconstruction, and plotting. Figures 4 and 5 use no overlap-weighted delay smoothing. |
| 2.8.3 | `score_uncertainty/` | Includes wrong matches with small residuals at W=1025 on repeated waveforms. |
| 2.8.4, 3.1 | `session_inference/`, `figures_28/`, `audit_v1/` through `audit_v3/` | Session variation, sample composition, and contact sensitivity. Anatomical labels require independent verification. |
| 3.2 | `examples/patchseq/`, `dataset_reporting/`, `figures_13/` | Follows sweeps, ISTD slopes, cell summaries, and gene associations. Enrichment was not resubmitted to Metascape. The associations do not establish a causal mechanism. |

Study directories in the table are relative to `examples/revision/studies/` unless a repository-level path is given. The repository layout was introduced in commit `9c998f2`, and the [known-delay validation](known_delay_validation.md) in commits `b214938` and `01637a1`.
