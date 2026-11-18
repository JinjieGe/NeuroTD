# 整合验证记录 / Integration validation

## 2026-11-18 revision evidence

| Result | Verification |
|---|---|
| Offline tests | 24 tests passed, including the single-command reconstruction of the nine tables and matched controls. |
| Tables S1–S9 | All nine generated CSVs matched the November 15 supplement byte for byte. S1 reports 0.000 ms for the noiseless NeuroTD result at W=513. |
| Regional contrasts (2.4.2) | Recomputed from 600 trial-by-region rows. A fresh run of 200,000 calibration datasets recovered the frozen SE multiplier 1.7519988157717314. All 96 rows of calibrated validation and challenge results agreed with the original rejection counts and interval coverage. The two out-of-library challenges remain in the simulation output. |
| Interference controls (2.4.3) | Reapplied Holm to 201,600 raw rank p-values in 2,800 families and checked 50,400 dynamic record-level RMSEs. Across seven scenarios, 5–14 of 200 constant-delay panels had at least one false rejection. White-noise dynamic controls had 99.83% detection and 0.152 ms mean RMSE. Repeated-waveform controls had 100% detection and 11.303 ms mean RMSE. |

The quick entry reads prior simulation outcomes. The independent regional calibration can also be run from its generators with `--repeat-calibration`. The matched-control entry starts after the original reference draws and delay estimation. See the [commands and outputs](revision-running.md).

## 2026-11-08 initial integration

### Checks performed on November 8

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

### Scope of the November 8 checks

- Numerical reproduction of every historical table or figure in a new software environment.
- A fresh run of the 1200 independent-noise pairs, 29379 complete-selection records, full session-level calibration, or 253440 SNR conditions.
- Availability of all raw NWB files and historical window checkpoints on another machine.
- General statistical calibration or biological validity beyond each module's stated assumptions.
- Comment 2.5 reliability analyses, which are excluded from this integration.

`source_manifest.csv` distinguishes recovered scripts from new integration utilities. `data_manifest.csv` distinguishes archived evidence from staged inputs. Full runs should save fresh manifests and results before those results are used in manuscript claims.
