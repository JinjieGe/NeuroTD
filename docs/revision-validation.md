# 整合验证记录 / Integration validation

Date: 2026-11-08. These checks validate the organized code bundle, **not a rerun of every manuscript experiment**.

## Verified in this integration

| Check | Result | Scope |
|---|---|---|
| Full offline unittest discovery | 23 tests passed | Repository layout/imports, known-delay audit, revision portability/scientific checks |
| Core estimator source | No diff from original `1a6c4f2` | `src/neurotd/time_varying_shift.py`, including the existing E |
| Known-delay launcher | Discrete smoke completed, exit 0 | Seed 0, noise 0/0.02, W=65/513; not the full grid |
| E identity | Passed | Independent score calculation and original function |
| Signal/model loading | Passed | Discrete and continuous clean signals; one candidate estimate each |
| Expanded-noise fast estimator | Passed | FFT/direct-MSE window centers and shifts identical on a known +30 sample fixture |
| Dynamic analysis report fixtures | Passed | Monte Carlo ties, independent splits, family statistic, missing-result rejection, exact binomial edges |
| Session inference pilot | Passed | All legacy scenarios at pilot size; no formal p-value claim from pilot |
| Processed Patch-seq inventory | Passed | 10637 rows, 1294 cells, sweep counts 1–56, median 7; three rows have fewer than two intervals |
| Archived figures | Five figure modules completed | 2.2, 2.3, 2.6, 2.8, and Reviewer 1/3; derived from saved tables or explicitly labeled schematics |
| Provenance | Hash manifests generated | Original and packaged source hashes; 64 small input/evidence files hash-matched to source bundles |

Commands used include:

```bash
PYTHONPATH=src MPLBACKEND=Agg python -m unittest discover -s tests -v
python examples/validate_known_delays.py --simulation discrete --smoke --output-dir /path/to/new-validation
python examples/revision/run.py run --workspace /path/to/new-dynamic --study dynamic_fixed --script selftest_report.py
python examples/revision/run.py run --workspace /path/to/new-stats --study session_inference --script code/run_analysis.py -- --pilot
```

All outputs were written outside the checkout. The figure-2.2 output was visually inspected. A missing bold variant of the Chinese font fell back to its regular weight in a separate bilingual plot; the figure commands completed. The original figure-1 docstring emits an existing invalid-escape warning during syntax checking; it does not affect these checks.

## Not established by these checks

- Numerical reproduction of every historical table or figure in a new software environment.
- A fresh run of the 1200 independent-noise pairs, 29379 complete-selection records, full session-level calibration, or 253440 SNR conditions.
- Availability of all raw NWB files and historical window checkpoints on another machine.
- General statistical calibration or biological validity beyond each module's stated assumptions.
- Comment 2.5 reliability analyses, which are excluded from this integration.

`source_manifest.csv` distinguishes recovered scripts from new integration utilities. `data_manifest.csv` distinguishes archived evidence from staged inputs. Full runs should save fresh manifests and results before those results are used in manuscript claims.
