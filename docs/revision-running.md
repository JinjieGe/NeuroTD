# Running the revision studies

Use Python 3.12 or later. Install the package and the additional study dependencies from the repository root.

```bash
python -m pip install -e .
python -m pip install -r examples/revision/requirements.txt
```

Choose a separate output workspace. The launcher records the command, code hashes, environment, and exit status. Historical source and packaged hashes are in `examples/revision/source_manifest.csv`.

## Rebuild the supplementary tables and final comparisons

```bash
python examples/revision/run.py evidence --workspace revision_runs/current
```

| Output under the workspace | Calculation |
|---|---|
| `supplement_tables/Table_S1.csv` through `Table_S9.csv` | S1–S2 use 66 noise-by-window combinations per benchmark. S3–S5 summarize independent noise repetitions. S6 uses trial summaries, S7 uses independent calibration validation, S8 uses the added-noise experiments, and S9 counts participants and sessions contributing to each regional comparison. |
| `session_inference/outputs/calibrated_welch_six_comparisons.csv` | Six regional contrasts from 600 trial-by-region summaries, with standard errors, effective degrees of freedom, adjusted p-values, and simultaneous intervals. S6 lists the three hippocampal comparisons first. |
| `dynamic_fixed/outputs/matched_control_summary.json` | Holm tests from archived rank p-values, panel false-positive rates, record-level detection rates, and delay RMSE by interference condition. |

This entry works offline. Run manifests are in `run_manifests/`.

## 2.2, 2.3, and 2.4.1. Selection and independent repetitions

```bash
python examples/revision/run.py run --workspace revision_runs/full --study score_uncertainty --script audit.py -- baseline
python examples/revision/run.py run --workspace revision_runs/full --study score_uncertainty --script audit.py -- run
python examples/revision/run.py run --workspace revision_runs/full --study score_uncertainty --script audit.py -- summarize
```

The baseline step checks the configuration, estimates runtime, and writes `baseline/comparison.csv`. The full run covers 1,200 independently generated noisy signal pairs and separate deterministic noiseless checks. The final summary requires complete checkpoints.

## 2.4.3. Delay dynamics and matched controls

```bash
python examples/revision/run.py run --workspace revision_runs/controls --study dynamic_fixed --script summarize_matched_controls.py
```

Each interference condition has 200 constant-delay panels. For each record, T is the mean squared deviation of the window delays from their mean (ms²) and S is the maximum absolute deviation (ms). Holm correction spans both measures for 36 records, or 72 tests in a panel. Any rejection in a constant panel counts as a familywise false positive. Dynamic panels report record-level detection and tracking RMSE. The reference bank used 2,999 draws per template. To check the original November 15 result files, append `-- --source-results /path/to/results --verify-endpoints`.

Earlier fixed-window simulations use the supplied clean signals.

```bash
python examples/revision/run.py run --workspace revision_runs/full --study dynamic_fixed --script simulate.py -- prepare
python examples/revision/run.py run --workspace revision_runs/full --study dynamic_fixed --script simulate.py -- run
python examples/revision/run.py run --workspace revision_runs/full --study dynamic_fixed --script simulate.py -- report
```

To include the full window-selection procedure in the null distribution:

```bash
python examples/revision/run.py run --workspace revision_runs/full --study dynamic_selection --script code/checks.py
python examples/revision/run.py run --workspace revision_runs/full --study dynamic_selection --script code/pipeline.py
python examples/revision/run.py run --workspace revision_runs/full --study dynamic_selection --script code/analyze.py
```

Run the consistency checks first. The historical scheduler monitors macOS resources. On other machines, check resource detection and worker count before starting the full run. Simulation calibration applies to the tested models.

## 2.4.2. Corrected regional comparisons

```bash
python examples/revision/run.py run --workspace revision_runs/stats --study session_inference --script code/calibrated_welch.py
python examples/revision/run.py run --workspace revision_runs/stats --study session_inference --script code/calibrated_welch.py -- --repeat-calibration
```

The first command recomputes six contrasts across four regions from one participant's three sessions of 50 trials each. It uses the frozen SE multiplier 1.7519988157717314, Welch-type degrees of freedom for each contrast, and Bonferroni correction across all six. The second command regenerates 200,000 calibration, 600,000 validation, and 40,000 out-of-library simulated datasets. Results go to `session_inference/outputs/`. For the earlier leave-one-session-out jackknife see `run_analysis.py`.

## 2.6. Dataset inventory

```bash
python examples/revision/run.py run --workspace revision_runs/inventory --study dataset_reporting --script audit_inputs.py
python examples/revision/run.py run --workspace revision_runs/inventory --study dataset_reporting --script audit_inputs.py -- --nwb-dir /path/to/dandiset_000574
```

The first command reads the published processed interval table. The second also reads NWB trial metadata without loading all voltages. The original raw Patch-seq processing and gene analysis are in `examples/patchseq/`. Source stimulation units and protocols require independent checking.

## 2.7.1. Added-noise experiments

Set `NEUROTD_RAW_DATA` to the folder containing `sub-01/`. The three NWB files must match the DANDI assets and checksums specified by the study.

```bash
export NEUROTD_RAW_DATA=/path/to/dandiset_000574
python examples/revision/run.py run --workspace revision_runs/snr --study snr_extension --script prepare_raw.py
python examples/revision/run.py run --workspace revision_runs/snr --study snr_extension --script extend_experiment.py -- prepare
python examples/revision/run.py run --workspace revision_runs/snr --study snr_extension --script extend_experiment.py -- run
python examples/revision/run.py run --workspace revision_runs/snr --study snr_extension --script interpret_results.py
```

Preparation stops if the files are missing. Append `-- --download` to the preparation command to download the large public NWB assets. The initial observational analysis also requires `audit_v1/checkpoints/windows_all_trials.npz` and `audit_v3/electrodes/pair_classification.csv`. The extended added-noise preparation runs independently of that checkpoint.

## 2.8.1. Delay search range

```bash
python examples/revision/run.py run --workspace revision_runs/range --study search_range --script run_search_range_ablation.py -- --workers 2
python examples/revision/run.py run --workspace revision_runs/range --study search_range --script analyze_search_range_ablation.py
```

The ablation retains both prespecified center-selection rules, all seeds, and failure cases. The baseline table comes from a historical local reproduction.

## Generate response figures

```bash
python examples/revision/run.py figures --workspace revision_runs/fig22 --study figures_22
python examples/revision/run.py figures --workspace revision_runs/fig23 --study figures_23
python examples/revision/run.py figures --workspace revision_runs/fig26 --study figures_26
python examples/revision/run.py figures --workspace revision_runs/fig28 --study figures_28
python examples/revision/run.py figures --workspace revision_runs/fig13 --study figures_13
```

Use a fresh workspace for each invocation. `figures_26` displays previously verified counts. `figures_13` includes a historical contact-sensitivity analysis.

## Quick checks

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_simulation_validation.py' -v
python -m unittest discover -s tests -p 'test_revision_bundle.py' -v
```

The tests cover syntax, scope, the E identity, both clean signal models, task counts, the fast and direct MSE implementations, processed-data counts, and workspace protection. See the [validation record](revision-validation.md) for the runs performed.
