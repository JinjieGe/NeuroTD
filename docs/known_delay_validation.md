# Validate window selection against known delays

See the [validation record](known_delay_validation_checks.md) for the runs performed and the conditions where selection failed.

## What the audit measures

- **E** is the implemented alignment score, `sqrt(mean(r_k))`, where `r_k` is the normalized residual for window k. Smaller values indicate closer waveform alignment. The submitted manuscript used a different aggregation for q in Equation (6).
- **Delay RMSE** compares estimated shifts with the simulated truth and converts samples to milliseconds using `1000/fs`.
- **Automatic selection** picks the candidate window with the smallest E using only the observations.
- **Oracle selection** picks the candidate with the smallest delay RMSE after the generating delays are revealed.
- **Excess RMSE** is automatic RMSE minus oracle RMSE. It measures the cost of choosing the window from the observations.

The audit records results after the original simulations finish their grids. It preserves the estimator, noise generation, default seeds, figure calculations, and FFT/CTW baseline implementations.

## Run the audit

Install the project with `python -m pip install -e .` from the repository root using Python 3.12 or later. Start with two noise levels and two windows. The continuous simulation also runs its original example trajectory.

```bash
python examples/validate_known_delays.py --simulation both --smoke
```

Run the full grid of eleven noise levels and six windows per model. The default seeds are 0 (discrete) and 2025 (continuous).

```bash
python examples/validate_known_delays.py --simulation both
```

For a multi-seed study, estimate runtime first, then specify the seeds before the main run.

```bash
python examples/validate_known_delays.py --simulation both --smoke --seeds 0 1 2
python examples/validate_known_delays.py --simulation both --seeds 0 1 2 3 4
```

Each run creates `validation_runs/<timestamp>/`. Use `--output-dir` to select a new directory. Existing destinations are protected. Figures are saved without a display, and each simulation writes progress to `run.log`. Older logs may call the implemented E "quality" or "q".

The launcher sets `NEUROTD_SIM_SEED`. Changing the discrete seed changes measurement noise. Changing the continuous seed also changes the random peaks and delay trajectory. All window sizes for a given seed share the same noise realization.

## Outputs

Each `discrete_seed_*` or `continuous_seed_*` folder contains a `known_delay_validation_*` directory.

| File | Contents |
|---|---|
| `candidate_metrics.csv` | RMSE, MAE, signed bias, and valid point count by method, noise level, and window. NeuroTD also reports E and the fraction of true delays outside its search range. |
| `window_selection.csv` | Selected and oracle window sizes and RMSEs, their difference and ratio, and whether the selected window attained oracle accuracy. |
| `delay_traces.csv.gz` | Center index, estimate, true delay, and pointwise error in samples for all four methods. Readable with pandas. |
| `selection_vs_truth.png` | Automatic and oracle RMSE for NeuroTD and the excess error from automatic selection. |
| `four_methods_known_delay_rmse.png` | Four methods compared at each fixed window size. |
| `summary.md` | Descriptive results, metric definitions, and limitations for that seed. |

The top-level `all_window_selections.csv` combines runs. `run_manifest.json` records the interpreter, dependencies, source hashes, Git revision, and completion status.

### Evaluation times and ties

- `native_centers` uses the valid centers returned by each method and window, following the original simulation tables.
- `shared_centers` uses only exact center indices present for every candidate and all four methods, without interpolation or extrapolation. Report `n_shared_centers` alongside results. The intersection may contain few or zero points. Search ranges and input segment conventions can still differ.

Selection uses full precision. Exact ties go to the smaller window, including ties in E. Delay RMSEs match the oracle within `rtol=1e-10, atol=1e-12 ms`. For zero oracle RMSE, the ratio is blank and the absolute excess is reported.

## Interpretation

1. Discrete truth comes from the nearest generated segment center. Continuous truth is evaluated at each output center. A window can span multiple true delays.
2. In `y(t)=x(t-Delta(t))`, the generating delay is defined on the y time axis. Obtaining the exact shift on the x reference axis requires inverting that time map. The audit keeps the paper's original convention.
3. Generating delays provide an evaluation target external to the residual score. Tuning on these simulations requires validation on fresh, prespecified settings or seeds.
4. Overlapping windows and noise levels within one seed are dependent. The audit reports descriptive errors without per-window significance tests or confidence intervals. Inference about physiological delays requires separate evidence.
5. Report the observed excess RMSE, including failures. Oracle performance describes a retrospective benchmark with known truth.

## Tests

```bash
python -m unittest discover -s tests -v
```

The tests cover units, true-delay errors, tie handling independent of true RMSE, zero denominators, missing and shared centers, nonfinite inputs, search bounds, and output protection.
